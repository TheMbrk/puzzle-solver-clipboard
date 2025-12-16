"""
Interactive Visual Puzzle Solver - Clipboard Mode
Just screenshot and paste! No file saving needed.
"""

from __future__ import annotations
import heapq
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Iterator, Set
import cv2
from PIL import ImageGrab, Image

# Board representation
Board = Tuple[int, ...]
GOAL: Board = tuple(range(1, 9)) + (0,)

# Precompute goal positions and neighbor moves
goal_positions: Dict[int, Tuple[int,int]] = {
    val: divmod(idx, 3) for idx, val in enumerate(GOAL)
}
neighbor_map: Dict[int, List[Tuple[int, str]]] = {}
for idx in range(9):
    i, j = divmod(idx, 3)
    for di, dj, move in [(0,1,'R'), (1,0,'D'), (0,-1,'L'), (-1,0,'U')]:
        ni, nj = i + di, j + dj
        if 0 <= ni < 3 and 0 <= nj < 3:
            neighbor_map.setdefault(idx, []).append((ni * 3 + nj, move))

# Precompute Manhattan distance lookup table
MANHATTAN_TABLE: Dict[Tuple[int, int], int] = {}
for tile in range(1, 9):
    goal_r, goal_c = goal_positions[tile]
    for pos in range(9):
        curr_r, curr_c = divmod(pos, 3)
        MANHATTAN_TABLE[(tile, pos)] = abs(curr_r - goal_r) + abs(curr_c - goal_c)


def get_clipboard_image() -> Optional[np.ndarray]:
    """Get image from clipboard."""
    try:
        img = ImageGrab.grabclipboard()
        if img is None:
            return None
        
        # Convert PIL image to OpenCV format
        img_array = np.array(img)
        
        # Convert RGB to BGR (OpenCV format)
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif img_array.shape[2] == 4:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                img_cv = img_array
        else:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        return img_cv
    except Exception as e:
        print(f"Error getting clipboard image: {e}")
        return None


def wait_for_clipboard_image(prompt: str) -> np.ndarray:
    """Wait for user to paste an image."""
    print(prompt)
    print("Waiting for image in clipboard...")
    
    # Try to get initial clipboard state
    initial_img = get_clipboard_image()
    
    while True:
        try:
            input("Press ENTER after you've taken the screenshot and it's in your clipboard...")
        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            exit(0)
        
        img = get_clipboard_image()
        
        if img is None:
            print("❌ No image found in clipboard!")
            print("   Please take a screenshot (it should automatically copy to clipboard)")
            print("   Then press ENTER")
            continue
        
        # Check if it's a different image than before
        if initial_img is not None and np.array_equal(img, initial_img):
            response = input("This looks like the same image. Use it anyway? (y/n): ").lower()
            if response != 'y':
                continue
        
        print(f"✓ Got image from clipboard! Size: {img.shape[1]}x{img.shape[0]}")
        return img


class TileRecognizer:
    """Handles image processing and tile recognition."""
    
    def __init__(self, puzzle_img: np.ndarray, goal_img: np.ndarray, debug=False):
        self.debug = debug
        self.puzzle_img = puzzle_img
        self.goal_img = goal_img
        
        # Extract tiles from both images
        self.puzzle_tiles = self._extract_tiles(self.puzzle_img)
        self.goal_tiles = self._extract_tiles(self.goal_img)
        
        print(f"Extracted {len(self.puzzle_tiles)} puzzle tiles")
        print(f"Extracted {len(self.goal_tiles)} goal tiles")
        
        if self.debug:
            self._save_debug_tiles()
    
    def _save_debug_tiles(self):
        """Save individual tiles for debugging."""
        import os
        os.makedirs("debug_tiles", exist_ok=True)
        
        for i, tile in enumerate(self.puzzle_tiles):
            cv2.imwrite(f"debug_tiles/puzzle_tile_{i}.png", tile)
        
        for i, tile in enumerate(self.goal_tiles):
            cv2.imwrite(f"debug_tiles/goal_tile_{i}.png", tile)
        
        # Save the full images too
        cv2.imwrite("debug_tiles/full_puzzle.png", self.puzzle_img)
        cv2.imwrite("debug_tiles/full_goal.png", self.goal_img)
        
        print("Debug files saved to debug_tiles/ directory")
    
    def _extract_tiles(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract 9 tiles from a 3x3 puzzle image."""
        h, w = image.shape[:2]
        tile_h, tile_w = h // 3, w // 3
        
        tiles = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * tile_h, (i + 1) * tile_h
                x1, x2 = j * tile_w, (j + 1) * tile_w
                tile = image[y1:y2, x1:x2].copy()
                tiles.append(tile)
        
        return tiles
    
    def _is_blank_tile(self, tile: np.ndarray) -> bool:
        """Determine if a tile is blank."""
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        std_val = np.std(gray)
        mean_val = np.mean(gray)
        
        if std_val < 5:
            return True
        if std_val < 15 and (mean_val > 240 or mean_val < 15):
            return True
        
        return False
    
    def _compute_tile_similarity(self, tile1: np.ndarray, tile2: np.ndarray) -> float:
        """Compute similarity between two tiles (lower = more similar)."""
        gray1 = cv2.cvtColor(tile1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(tile2, cv2.COLOR_BGR2GRAY)
        
        if gray1.shape != gray2.shape:
            min_h = min(gray1.shape[0], gray2.shape[0])
            min_w = min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (min_w, min_h))
            gray2 = cv2.resize(gray2, (min_w, min_h))
        
        diff = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
        return diff / 255.0
    
    def recognize_board_state(self) -> Board:
        """Recognize the current board state."""
        board = [0] * 9
        
        # Find blank tiles
        puzzle_blank = None
        goal_blank = None
        
        for i, tile in enumerate(self.puzzle_tiles):
            if self._is_blank_tile(tile):
                puzzle_blank = i
                print(f"Found blank tile at puzzle position {i}")
                break
        
        for i, tile in enumerate(self.goal_tiles):
            if self._is_blank_tile(tile):
                goal_blank = i
                print(f"Found blank tile at goal position {i}")
                break
        
        if puzzle_blank is None:
            print("WARNING: No blank tile detected in puzzle")
            variances = [np.std(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)) for t in self.puzzle_tiles]
            puzzle_blank = variances.index(min(variances))
            print(f"  Using position {puzzle_blank} as blank (lowest variance)")
        
        if goal_blank is None:
            print("WARNING: No blank tile detected in goal, assuming position 8")
            goal_blank = 8
        
        board[puzzle_blank] = 0
        
        # Build similarity matrix
        similarity_matrix = np.full((9, 9), 999.0)
        
        for p_idx in range(9):
            if p_idx == puzzle_blank:
                continue
            
            for g_idx in range(9):
                if g_idx == goal_blank:
                    continue
                
                similarity = self._compute_tile_similarity(
                    self.puzzle_tiles[p_idx],
                    self.goal_tiles[g_idx]
                )
                similarity_matrix[p_idx][g_idx] = similarity
        
        # Greedy matching
        used_goal_positions = {goal_blank}
        used_puzzle_positions = {puzzle_blank}
        
        while len(used_puzzle_positions) < 9:
            best_score = float('inf')
            best_p = None
            best_g = None
            
            for p_idx in range(9):
                if p_idx in used_puzzle_positions:
                    continue
                
                for g_idx in range(9):
                    if g_idx in used_goal_positions:
                        continue
                    
                    if similarity_matrix[p_idx][g_idx] < best_score:
                        best_score = similarity_matrix[p_idx][g_idx]
                        best_p = p_idx
                        best_g = g_idx
            
            if best_p is None:
                break
            
            if best_g < goal_blank:
                tile_id = best_g + 1
            else:
                tile_id = best_g
            
            board[best_p] = tile_id
            used_puzzle_positions.add(best_p)
            used_goal_positions.add(best_g)
            
            print(f"Matched: Puzzle pos {best_p} = Goal tile {tile_id} (score={best_score:.3f})")
        
        return tuple(board)


@dataclass(order=True)
class PuzzleState:
    f: int
    neg_g: int
    board: Board = field(compare=False)
    g: int = field(default=0, compare=False)
    h: int = field(default=0, compare=False)
    move: Optional[str] = field(default=None, compare=False)
    zero_idx: int = field(default=0, compare=False)

    def __post_init__(self):
        if self.h == 0:
            self.h = sum(
                MANHATTAN_TABLE[(val, i)]
                for i, val in enumerate(self.board) if val != 0
            )
        self.f = self.g + self.h
        self.neg_g = -self.g
        if self.zero_idx == 0:
            self.zero_idx = self.board.index(0)

    def neighbors(self) -> Iterator[PuzzleState]:
        for new_idx, move in neighbor_map[self.zero_idx]:
            new_board = list(self.board)
            moved_tile = new_board[new_idx]
            new_board[self.zero_idx], new_board[new_idx] = moved_tile, 0
            
            h_delta = (
                MANHATTAN_TABLE[(moved_tile, self.zero_idx)] - 
                MANHATTAN_TABLE[(moved_tile, new_idx)]
            )
            new_h = self.h + h_delta
            
            yield PuzzleState(
                board=tuple(new_board),
                g=self.g + 1,
                h=new_h,
                move=move,
                f=0,
                neg_g=0,
                zero_idx=new_idx
            )


def is_solvable(board: Board) -> bool:
    inv = sum(
        1
        for i in range(9)
        for j in range(i + 1, 9)
        if board[i] and board[j] and board[i] > board[j]
    )
    return inv % 2 == 0


def reconstruct_path(came_from: Dict[Board, Tuple[Board, str]], end: Board) -> List[str]:
    path: List[str] = []
    cur = end
    while cur in came_from:
        prev, move = came_from[cur]
        path.append(move)
        cur = prev
    return list(reversed(path))


def a_star(start: Board) -> Optional[List[str]]:
    if not is_solvable(start):
        return None
    
    came_from: Dict[Board, Tuple[Board, str]] = {}
    closed: Set[Board] = set()
    
    frontier = [PuzzleState(f=0, neg_g=0, board=start)]
    g_scores: Dict[Board, int] = {start: 0}
    
    while frontier:
        state = heapq.heappop(frontier)
        
        if state.board in closed:
            continue
            
        if state.board == GOAL:
            return reconstruct_path(came_from, state.board)
        
        closed.add(state.board)
        
        for neigh in state.neighbors():
            if neigh.board in closed:
                continue
            
            if neigh.g < g_scores.get(neigh.board, float('inf')):
                came_from[neigh.board] = (state.board, neigh.move)
                g_scores[neigh.board] = neigh.g
                heapq.heappush(frontier, neigh)
    
    return None


def main() -> None:
    print("=" * 60)
    print("VISUAL PUZZLE SOLVER - CLIPBOARD MODE")
    print("=" * 60)
    print()
    print("This mode grabs images directly from your clipboard!")
    print("Just screenshot and paste - no need to save files.")
    print()
    print("How to use:")
    print("  1. Take a screenshot of the GOAL/HINT (Win+Shift+S / Cmd+Shift+4)")
    print("  2. Press ENTER here")
    print("  3. Take a screenshot of the CURRENT PUZZLE")
    print("  4. Press ENTER here")
    print("  5. Get your solution!")
    print()
    print("-" * 60)
    
    # Get goal image from clipboard
    goal_img = wait_for_clipboard_image(
        "\n📸 STEP 1: Screenshot the GOAL/HINT (the solved puzzle)"
    )
    
    # Get puzzle image from clipboard
    puzzle_img = wait_for_clipboard_image(
        "\n📸 STEP 2: Screenshot the CURRENT PUZZLE (what you need to solve)"
    )
    
    # Ask if user wants debug mode
    print()
    debug = input("Enable debug mode? (saves images for inspection) (y/n): ").lower() == 'y'
    
    # Recognize tiles
    print()
    print("=" * 60)
    print("ANALYZING IMAGES")
    print("=" * 60)
    recognizer = TileRecognizer(puzzle_img, goal_img, debug=debug)
    start = recognizer.recognize_board_state()
    
    print(f"\nRecognized board state: {start}")
    print("\nBoard visualization:")
    for i in range(3):
        row = start[i*3:(i+1)*3]
        print("  " + " ".join(str(x) if x != 0 else "_" for x in row))
    
    # Validate
    if start.count(0) != 1:
        print(f"\n❌ ERROR: Invalid board state")
        print(f"   Found {start.count(0)} blank tiles (expected 1)")
        return
    
    if sorted(start) != list(range(9)):
        print(f"\n❌ ERROR: Invalid board state")
        print(f"   Tiles should be 0-8, got: {sorted(start)}")
        return
    
    # Solve
    print()
    print("=" * 60)
    print("SOLVING PUZZLE")
    print("=" * 60)
    
    t0 = time.perf_counter()
    sol = a_star(start)
    t1 = time.perf_counter()
    
    print()
    if sol is None:
        print("❌ No solution found (puzzle is unsolvable)")
        inv_count = sum(1 for i in range(9) for j in range(i+1,9) 
                       if start[i] and start[j] and start[i]>start[j])
        print(f"   Inversion count: {inv_count} (must be even to be solvable)")
    else:
        print("=" * 60)
        print(f"✓ SOLVED IN {len(sol)} MOVES!")
        print("=" * 60)
        print()
        print("Move sequence:")
        print(f"  {' '.join(sol)}")
        print()
        print(f"Time: {t1 - t0:.3f} seconds")
        print()
        print("Move guide:")
        print("  U = Move tile UP    (blank moves down)")
        print("  D = Move tile DOWN  (blank moves up)")
        print("  L = Move tile LEFT  (blank moves right)")
        print("  R = Move tile RIGHT (blank moves left)")
        print()
        print("=" * 60)
    
    input("\nPress ENTER to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress ENTER to exit...")

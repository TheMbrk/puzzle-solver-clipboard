# Clipboard Mode - Quick Guide 📋

## The Easiest Way to Solve Puzzles!

No file saving, no cropping, just screenshot and go!

## 🚀 How to Use

### 1. Run the Script

```bash
python puzzle_solver_clipboard.py
```

### 2. Screenshot the Goal/Hint

- **Windows**: Press `Win + Shift + S`, select the goal puzzle
- **Mac**: Press `Cmd + Shift + 4`, select the goal puzzle
- **Linux**: Use your screenshot tool

The screenshot is automatically in your clipboard!

### 3. Press ENTER

The script is waiting for you. Just press ENTER.

### 4. Screenshot the Current Puzzle

Take another screenshot of what you need to solve.

### 5. Press ENTER Again

Done! The solution appears instantly.

## 📺 Example Session

```
============================================================
VISUAL PUZZLE SOLVER - CLIPBOARD MODE
============================================================

📸 STEP 1: Screenshot the GOAL/HINT (the solved puzzle)
Waiting for image in clipboard...
Press ENTER after you've taken the screenshot...

✓ Got image from clipboard! Size: 300x300

📸 STEP 2: Screenshot the CURRENT PUZZLE
Waiting for image in clipboard...
Press ENTER after you've taken the screenshot...

✓ Got image from clipboard! Size: 300x300

============================================================
ANALYZING IMAGES
============================================================
Found blank tile at puzzle position 7
Found blank tile at goal position 8
Matched: Puzzle pos 0 = Goal tile 2 (score=0.000)
...

============================================================
SOLVING PUZZLE
============================================================

============================================================
✓ SOLVED IN 12 MOVES!
============================================================

Move sequence:
  R D L U R D L U R D L U

Time: 0.234 seconds
```

## 💡 Tips

### Getting Good Screenshots

✅ **Include just the puzzle** - Don't need borders or UI  
✅ **Make sure it's 3×3** - Should have 9 tiles  
✅ **Clear and sharp** - Not blurry  
✅ **Good lighting** - Can see all tiles clearly  

### Common Issues

**"No image found in clipboard"**
- Make sure you used the screenshot tool (not just Print Screen)
- On Windows, use Win+Shift+S (Snipping Tool)
- Screenshot should auto-copy to clipboard

**"Invalid board state"**
- Check that your screenshots show complete 3×3 grids
- Make sure one tile is clearly blank/empty
- Try the debug mode to see what was detected

### Debug Mode

If tiles aren't being recognized correctly:

When prompted, type `y` for debug mode. This saves:
- `debug_tiles/full_puzzle.png` - What the solver saw
- `debug_tiles/full_goal.png` - The goal image
- `debug_tiles/puzzle_tile_0.png` through `puzzle_tile_8.png`
- `debug_tiles/goal_tile_0.png` through `goal_tile_8.png`

## 🎮 Perfect For

- **Mobile games** - Screenshot on your phone, solve on PC
- **Web games** - Quick screenshot, instant solution
- **Homework** - Fast help with sliding puzzles
- **Speedrunning** - Optimize your puzzle solving

## ⚡ Alternative Modes

If you prefer working with saved files:

```bash
# Single screenshot with both puzzles
python visual_puzzle_solver.py screenshot.png

# Separate files
python visual_puzzle_solver.py puzzle.png goal.png
```

## 🔧 Requirements

Make sure you have the required libraries:

```bash
pip install numpy opencv-python Pillow
```

## ❓ FAQ

**Q: Do I need to save my screenshots?**  
A: Nope! Just screenshot and the script grabs it from clipboard.

**Q: What if I screenshot the wrong thing?**  
A: Just Ctrl+C to cancel and run the script again.

**Q: Can I use this on Mac/Linux?**  
A: Yes! Works on all platforms with clipboard support.

**Q: Does it work with any puzzle game?**  
A: Works with any 3×3 sliding puzzle that shows a goal/hint.

## 🎯 Pro Tips

1. **Keep the window visible** - So you can see when to press ENTER
2. **Screenshot quickly** - Goal first, then puzzle
3. **Use keyboard shortcuts** - Much faster than mouse menus
4. **Check the visualization** - Make sure the board was read correctly
5. **Save hard puzzles** - Enable debug mode to keep the screenshots

## 🚫 Troubleshooting

### Screenshots aren't being captured

**Windows**: 
- Use Win+Shift+S (Snipping Tool) - NOT just Print Screen
- Print Screen doesn't always copy to clipboard

**Mac**:
- Use Cmd+Shift+4 for region select
- Use Cmd+Shift+3 for full screen
- Hold Control while screenshotting to copy to clipboard

**Linux**:
- Use your distribution's screenshot tool
- Make sure it copies to clipboard (check settings)

### Wrong tiles being matched

- Take clearer screenshots
- Make sure each tile is visually distinct
- Ensure good lighting/contrast
- Try debug mode to see what was captured

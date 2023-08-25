import mouse
import tkinter

print(mouse.get_position())

# mouse.click('left')
# mouse.click('right')
# mouse.click('middle')

# print(mouse.is_pressed("right"))

# mouse.drag(0, 0, 100, 100, absolute=False, duration=0.1)

# mouse.move(100, 100, absolute=False, duration=0.1)

# # make a listener when button is clicked
# mouse.on_click(lambda: print("Left Button clicked."))
# mouse.on_right_click(lambda: print("Right Button clicked."))
# # remove the listeners when you want
# mouse.unhook_all()

# # scroll down
mouse.wheel(-1)

print(mouse.get_position())

# # scroll up
mouse.wheel(10)

print(mouse.get_position())


root = tkinter.Tk()
root.withdraw()
WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
print(WIDTH, HEIGHT)
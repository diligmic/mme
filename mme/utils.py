import os, select, sys

if os.name == 'nt':
    import msvcrt

def heardEnter():
    # Listen for the user pressing ENTER

    if os.name == 'nt':
        if msvcrt.kbhit():
            if msvcrt.getch() == b"q":
                print("Quit key pressed, saving the model...")
                return True
        else:
            return False
    else:
        i, o, e = select.select([sys.stdin], [], [], 0.0001)
        for s in i:
            if s == sys.stdin:
                input = sys.stdin.readline()
                return True
        return False
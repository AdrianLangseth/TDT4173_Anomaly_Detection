num_tabs = 0

def __print(*args, **kwargs):
    print("\t"*num_tabs, end="")
    print(*args, **kwargs)


def print_area_start(area_name):
    print()
    __print("-"*5, area_name, "-"*5)
    print()
    global num_tabs 
    num_tabs += 1

def print_area_content(content):
    __print(content)

def print_area_end():
    global num_tabs
    num_tabs = max(0, num_tabs - 1)
    __print("-"*25)

def print_area(area_name, *contents):
    print_area_start(area_name)
    all(print_area_content(content) for content in contents)
    print_area_end()

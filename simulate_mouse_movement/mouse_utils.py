# 1 -> index, 2 -> middle, 3 -> ring, d(i,j) -> distance b/w i and j, m -> threshold min value, n -> threshold max value
# 1,2,3 -> drag | d(1,2) < m and d(2,3) < m 
# 1,2 -> move | d(1,2) < m and d(2,3) > n and d(1,3) > n
# 1 -> left click | d(1,2) > n and d(1,3) > n
# 2 -> right click | d(2,3) > n and d(2,1) > n
# m = 0.1, n = 0.2

import mouse
import time

# def if_drag(dist, minV, maxV, z1, z2, z3, z4, th, depth_factor):
#     # return dist["d1"] < minV and dist["d2"] < minV 
#     return dist["d1"] < minV and dist["d2"] < minV and \
#     th[0] < minV and th[1] < minV
#     # return dist["d1"] < minV and dist["d2"] < minV and \
#     #     th[0] < minV and th[1] < minV and \
#     #     0.7 < min(abs(z1-z4)/abs(z1-z2), abs(z1-z2)/abs(z1-z4)) < 1.3
#     # return dist["d1"] < minV  and th[0] < minV and th[1] < minV\
#     #     and (abs(z1-z2) + abs(z1-z3) <= 0.8*max(abs(z2), abs(z3)))
#     # return (abs(z1-z3) + abs(z4-z3) > (depth_factor/1)*abs(z1-z4)) \
#     #     or dist["d1"] < minV  and th[0] < minV and th[1] < minV

def if_drag(dist, minV, maxV, z1, z2, z3, z4, th, depth_factor):
    return dist["d1"] < minV and dist["d2"] < minV and dist["d3"] < minV

def drag_cursor(x1, y1, x2, y2, width_factor, height_factor, width, height, speed, minPixel):
    x, y = int((x1-x2)*width_factor*width), int((y1-y2)*height_factor*height)
    dur = (((x**2 + y**2)**0.5)*speed)/(100*(2**0.5))
    if (x**2 + y**2)**0.5 > minPixel:
        mouse.drag(0, 0, -x, y, absolute=False, duration=dur)


def if_move(dist, maxV, z1, z2, z3, depth_factor):
    return (abs(z1-z2) + abs(z1-z3) > (depth_factor*2.5)*abs(z2-z3)) or (dist["d1"] > maxV and dist["d3"] > maxV)
    
def move_cursor(x1, y1, x2, y2, width_factor, height_factor, width, height, speed, minPixel):
    x, y = int((x1-x2)*width_factor*width), int((y1-y2)*height_factor*height)
    dur = (((x**2 + y**2)**0.5)*speed)/(100*(2**0.5))
    if (x**2 + y**2)**0.5 > minPixel:
        mouse.move(-x, y, absolute=False, duration=dur)


def if_left_click(dist, minV, maxV, z1, z2, z3, depth_factor):
    return (abs(z1-z3) + abs(z2-z3) > depth_factor*abs(z1-z2)) \
        or dist["d1"] < minV and dist["d2"] > maxV and dist["d3"] > maxV


def left_click(sleep_time):
    mouse.click('left')
    time.sleep(sleep_time)


def if_right_click(dist, maxV, z1, z2, z3, depth_factor):
    return (abs(z1-z2) + abs(z2-z3) > depth_factor*abs(z1-z3)) or dist["d1"] > maxV and dist["d2"] > maxV
    

def right_click(sleep_time):
    mouse.click('right')
    time.sleep(sleep_time)


def if_pause(dist, x1, y1, x2, y2, minV, maxV):
    d = ((x1-x2)**2 + (y1-y2)**2)**0.5
    return d < minV and dist["d1"] > maxV and dist["d3"] > maxV and dist["d2"] < minV

# z1 -> index, z2 -> middle, z3 -> ring, z4 -> thumb
# th[0] -> dist(thumb, index), th[1] -> dist(thumb, middle), th[2] -> dist(thumb, ring)
def check_activity(dist, minV, maxV, z1, z2,z3, z4, th, depth_factor):
    if if_move(dist, maxV, z1, z2, z3, depth_factor):
        return "move"
    elif if_drag(dist, minV, maxV, z1, z2, z3, z4, th, depth_factor):
        return "drag"
    elif if_left_click(dist, minV, maxV, z1, z2, z3, depth_factor):
        return "left_click"
    elif if_right_click(dist, maxV, z1, z2, z3, depth_factor):
        return "right_click"
    return


def do_activity(name, x1, y1, x2, y2, width_factor, height_factor, width, height, speed, sleep_time, minPixel=None):
    if name == "move":
        move_cursor(x1, y1, x2, y2, width_factor, height_factor, width, height, speed, minPixel)
    elif name == "drag":
        drag_cursor(x1, y1, x2, y2, width_factor, height_factor, width, height, speed, minPixel)
    elif name == "left_click":
        left_click(sleep_time)
    elif name == "right_click":
        right_click(sleep_time)


def if_skip_frame(x1, y1, x2, y2, width_factor, height_factor, width, height, skip_pixel):
    x, y = int((x1-x2)*width_factor*width), int((y1-y2)*height_factor*height)
    return (x**2 + y**2)**0.5 < skip_pixel

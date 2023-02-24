import time

import mouse
import numpy as np
import win32con
import win32gui
import win32ui
from pynput import keyboard

from detector import Detector


def window_prepare():
    hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    # mfcDC创建可兼容的DC
    save_dc = mfc_dc.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    save_bit_map = win32ui.CreateBitmap()
    # 获取监控器信息
    width = 640
    height = 360
    # 为bitmap开辟空间
    save_bit_map.CreateCompatibleBitmap(mfc_dc, width, height)
    return save_bit_map, save_dc, mfc_dc, width, height


def window_capture(save_bit_map, save_dc, mfc_dc, width, height):
    # 高度saveDC，将截图保存到saveBitmap中
    save_dc.SelectObject(save_bit_map)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (640, 360), win32con.SRCCOPY)
    signed_ints_array = save_bit_map.GetBitmapBits(True)
    img = np.frombuffer(signed_ints_array, dtype='uint8')
    img.shape = (height, width, 4)
    img0 = img[:, :, :3]
    return img0


def aiming(save_bit_map, save_dc, mfc_dc, width, height):
    # 读取每帧图片
    a1 = time.time()
    im = window_capture(save_bit_map, save_dc, mfc_dc, width, height)
    a2 = time.time()
    bboxes = detector.detect(im)
    a3 = time.time()
    # 如果画面中有bbox，即detector探测到待检测对象
    if len(bboxes) > 0:

        box_index = detector.closest(bboxes)
        check_point_x = int(bboxes[box_index][0] + ((bboxes[box_index][2] - bboxes[box_index][0]) * 0.5)) + 640
        check_point_y = int(bboxes[box_index][1] + ((bboxes[box_index][3] - bboxes[box_index][1]) * 0.5)) + 360
        move_x = int(1.4 * (check_point_x - 960))
        move_y = int(1.4 * (check_point_y - 540))
        mouse.move(move_x, move_y, absolute=False, duration=0)
        # time.sleep(0.0001)
        a4 = time.time()
        # print(a4-a1)
        # print(a2-a1)
        # print(a3-a2)
        # print(a4-a3)
        pass
    else:
        pass
    pass

    pass


mode = False


def on_press(key):
    global mode
    if key == keyboard.Key.f11:
        mode = not mode


def on_release(key):
    """松开按键时执行。"""
    # if key == keyboard.Key.esc:
    #     pass
    pass


if __name__ == '__main__':
    # 初始化 yolov5
    detector = Detector()
    saveBitMap, saveDC, mfcDC, w, h = window_prepare()
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    while True:
        if mode:
            aiming(saveBitMap, saveDC, mfcDC, w, h)
        time.sleep(0.00001)
    # while True:
    #     if keyboard0.is_pressed('tab'):
    #         mode = 1
    #     if keyboard0.is_pressed('alt'):
    #         mode = 0
    #     if mode:
    #         aiming()
    #     time.sleep(0.00001)

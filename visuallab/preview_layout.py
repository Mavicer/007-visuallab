"""One native window with a fixed 2:1 left/right layout.

Only display copies are resized; inference and callbacks keep camera pixels.
"""
import cv2
import numpy as np
import ctypes
import platform

PREVIEW_WINDOW = "VisualLab - Workspace"
INFO_HEIGHT = 140


def display_bounds(max_width: int = 1800, max_height: int = 960) -> tuple[int,int]:
    """Use macOS desktop bounds, reserving title bar/menu/Dock space.

    Display pixels and window coordinates can differ on Retina displays.
    A headless/sandboxed session may expose no display; use a safe fallback.
    """
    if platform.system() == "Darwin":
        class Point(ctypes.Structure):
            _fields_ = [("x",ctypes.c_double),("y",ctypes.c_double)]
        class Size(ctypes.Structure):
            _fields_ = [("width",ctypes.c_double),("height",ctypes.c_double)]
        class Rect(ctypes.Structure):
            _fields_ = [("origin",Point),("size",Size)]
        try:
            cg=ctypes.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
            cg.CGMainDisplayID.restype=ctypes.c_uint32
            cg.CGDisplayBounds.argtypes=[ctypes.c_uint32]
            cg.CGDisplayBounds.restype=Rect
            size=cg.CGDisplayBounds(cg.CGMainDisplayID()).size
            if size.width >= 640 and size.height >= 480:
                return min(max_width,int(size.width)-80),min(max_height,int(size.height)-130)
        except (OSError,AttributeError):
            pass
        return min(max_width,1440),min(max_height,800)
    return max_width,max_height


def preview_size(shape: tuple[int, ...], max_width: int = 1800,
                 max_height: int = 960) -> tuple[int, int]:
    """Keep original-aspect image space plus a dedicated information band."""
    height, width = shape[:2]
    scale = min(max_width/(1.5*width), max(2,max_height-INFO_HEIGHT)/height)
    small_w = max(1, int(width*scale/2))
    small_h = max(1, int(height*scale/2))
    return small_w*3, small_h*2+INFO_HEIGHT


def _panel(image: np.ndarray, width: int, height: int, title: str,
           *, binary: bool = False, header_height: int = 24) -> np.ndarray:
    """Letterbox a complete image under its header; never stretch/crop it."""
    canvas = np.full((height,width,3),(18,21,26),np.uint8)
    header = min(header_height,max(0,height-1))
    h,w = image.shape[:2]
    ratio = min(width/w,max(1,height-header)/h)
    rw,rh = max(1,round(w*ratio)),max(1,round(h*ratio))
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR
    resized = cv2.resize(image,(rw,rh),interpolation=interpolation)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR)
    x,y = (width-rw)//2,header+(height-header-rh)//2
    canvas[y:y+rh,x:x+rw] = resized
    if header:
        cv2.rectangle(canvas,(0,0),(width-1,header-1),(34,39,47),-1)
        cv2.putText(canvas,title,(10, min(18,max(1,header-7))),cv2.FONT_HERSHEY_SIMPLEX,.52,
                    (220,230,240),1,cv2.LINE_AA)
    return canvas


def compose_preview(gray: np.ndarray, binary: np.ndarray, marked: np.ndarray, *,
                    info_lines: tuple[str, ...] | list[str] = (),
                    max_width: int = 1800, max_height: int = 960) -> np.ndarray:
    """Left result; upper-right gray; lower-right binary, exactly aligned."""
    width,height = preview_size(marked.shape,max_width,max_height)
    sw,sh = width//3,height//2
    main = _panel(marked,2*sw,2*sh,"RESULT",header_height=INFO_HEIGHT)
    right = np.vstack((_panel(gray,sw,sh,"GRAY"),
                       _panel(binary,sw,sh,"BINARY",binary=True)))
    if info_lines:
        # English labels fit OpenCV's built-in font. Model names are bounded
        # to the main panel, so a long path cannot cover the right column.
        lines = [str(line).encode('ascii',errors='replace').decode() for line in info_lines[:5]]
        font,scale = cv2.FONT_HERSHEY_SIMPLEX,.52
        clipped=[]
        for line in lines:
            while line and cv2.getTextSize(line,font,scale,1)[0][0]>2*sw-20:
                line=line[:-1]
            clipped.append(line)
        for index,line in enumerate(clipped):
            cv2.putText(main,line,(12,42+index*22),font,scale,(210,230,220),1,cv2.LINE_AA)
    canvas = np.hstack((main,right))
    cv2.line(canvas,(2*sw,0),(2*sw,height-1),(65,72,85),1)
    cv2.line(canvas,(2*sw,sh),(width-1,sh),(65,72,85),1)
    return canvas

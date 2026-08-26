from .env import PushTEnv
import cv2 as cv
import numpy as np


def main() -> None:
    """以持续按键控制方式观察 Pymunk Push-T 的平移与旋转。"""
    t_env = PushTEnv()
    t_env.reset()
    while True:
        action = np.zeros(2, dtype=np.float32)
        key = cv.waitKey(16) & 0xFF
        if key == ord("d"):
            action[0] = 1.0
        elif key == ord("a"):
            action[0] = -1.0
        elif key == ord("s"):
            action[1] = 1.0
        elif key == ord("w"):
            action[1] = -1.0
        elif key == ord("q"):
            break
        img, stat = t_env.step(action)
        view = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.putText(
            view,
            f"angle: {np.degrees(stat.object_angle):.1f} deg",
            (8, 24),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )
        cv.imshow("Push-T", view)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

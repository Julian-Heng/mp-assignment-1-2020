#!/usr/bin/env python3

import cv2

from abc import ABC
from multiprocessing import Pool, cpu_count

from .image import Image
from .utils import use_dir


class Task(ABC):

    def __init__(self, img, num):
        super(Task, self).__init__()

        self._img = img
        self._num = num
        self._fname = img.filename

        self._current_variant = None
        self._current_action = None
        self._img.print_write_callback = self._log_write

        # Default variants
        self._variants = {
            "original": lambda i: i,

            # Scaling
            "0.1x_scale": lambda i: i.resize(factor=0.1),
            "0.5x_scale": lambda i: i.resize(factor=0.5),
            "1.5x_scale": lambda i: i.resize(factor=1.5),
            "2.0x_scale": lambda i: i.resize(factor=2.0),
            "mixed_scale": lambda i: i.resize(factor=(2.5, 0.5)),
            "extreme_mixed_scale": lambda i: i.resize(factor=(10.0, 0.1)),

            # Rotation
            "20_rotate": lambda i: i.rotate(angle=20.0),
            "45_rotate": lambda i: i.rotate(angle=45.0),
            "90_rotate": lambda i: i.rotate(angle=90.0),
            "180_rotate": lambda i: i.rotate(angle=180.0),

            # Misc
            "increase_brightness": lambda i: i.brightness(value=100),
            "decrease_brightness": lambda i: i.brightness(value=-100),
            "gaussian_15": lambda i: i.gaussian(ksize=(15, 15)),
            "gaussian_5": lambda i: i.gaussian(ksize=(5, 5)),
            "normalized": lambda i: i.normalize(),
        }

        # Default actions
        self._actions = {
            "write": lambda i, name: i.write(append=name),
        }

    def run(self):
        """ Create the image variations and do the actions """
        count = 0
        total = len(self._variants) * len(self._actions)
        for variant_name, do_variant in self._variants.items():
            self._current_variant = variant_name
            for action_name, do_action in self._actions.items():
                self._current_action = action_name
                count += 1
                self._log(count, total)

                self._img.reset()
                do_variant(self._img)
                do_action(self._img, variant_name)

    def _log(self, count, total):
        def _colorise_bracket(text, color):
            fmt = "{{}}[{{}}{}{{}}]{{}}"
            fmt = fmt.format(text)
            fmt = fmt.format("\033[1m\033[37m", color, "\033[37m", "\033[0m")
            return fmt

        task = _colorise_bracket(self._num, "\033[32m")
        fname = _colorise_bracket(self._fname, "\033[34m")
        variant = _colorise_bracket(self._current_variant, "\033[34m")
        percent = _colorise_bracket(
            "{:.02f}%".format((count / total) * 100), "\033[34m"
        )
        action = ":\n {{}}=>{{}} {{}}Performing{{}} '{}'".format(
            self._current_action
        )
        action = action.format("\033[33m\033[1m", "\033[0m", "\033[1m", "\033[0m")
        print("".join([task, fname, variant, percent, action]))

    def _log_write(self, fname):
        fmt = " {{}}=>{{}} {{}}Writing{{}} '{}'"
        fmt = fmt.format(fname)
        fmt = fmt.format("\033[33m\033[1m", "\033[0m", "\033[1m", "\033[0m")
        print(fmt.format(fname))


class Task1(Task):

    def __init__(self, img):
        super(Task1, self).__init__(img, 1)

        self._actions = {
            **self._actions,
            "histogram": lambda i, name: i.histogram(append=name),
            "harris": lambda i, name: i.harris(
                block_size=3, kernel_size=3, k=0.04, threshold=0.05, color="blue"
            ).write(append="{}_harris".format(name)),
            "sift keypoints": lambda i, name: i.sift_keypoints().write(
                append="{}_sift".format(name)
            ),
        }


class Task1Dugong(Task1):

    def __init__(self, img):
        super(Task1Dugong, self).__init__(img)

        self._variants = {
            **self._variants,
            "45_rotate_cropped": lambda i: i.rotate(angle=45.0).center(size=250),
        }


class Task2(Task):

    def __init__(self, img):
        super(Task2, self).__init__(img, 2)

        self._actions = {
            **self._actions,
            "hog": lambda i, name: i.hog(append=name),
            "sift descriptors": lambda i, name: i.sift_descriptors().write(
                append="{}_sift".format(name)
            ),
        }


class Task2Diamond(Task2):

    def __init__(self, img):
        super(Task2Diamond, self).__init__(img)

        self._variants = {
            **self._variants,
            "20_rotate_cropped": lambda i: i.rotate(angle=20.0).center(size=50),
            "45_rotate_cropped": lambda i: i.rotate(angle=45.0).center(size=50),
        }


class Task2Dugong(Task2):

    def __init__(self, img):
        super(Task2Dugong, self).__init__(img)

        self._variants = {
            **self._variants,
            "20_rotate_cropped": lambda i: i.rotate(angle=20.0).center(size=250),
            "45_rotate_cropped": lambda i: i.rotate(angle=45.0).center(size=250),
        }


class Task3(Task):

    def __init__(self, img):
        super(Task3, self).__init__(img, 3)

        self._variants = {
            "original": lambda i: i,
        }


class DiamondTask3(Task3):

    def __init__(self, img):
        super(DiamondTask3, self).__init__(img)

        self._actions = {
            **self._actions,
            "binary": lambda i, name: (
                i.threshold(threshold=127, threshold_type=cv2.THRESH_BINARY_INV).write(
                    append="{}_binary".format(name)
                )
            ),
            "connected_components": lambda i, name: (
                i.threshold(
                    threshold=127, threshold_type=cv2.THRESH_BINARY_INV
                ).connected_components(count=True, append=name)
            ),
        }


class DugongTask3(Task3):

    def __init__(self, img):
        super(DugongTask3, self).__init__(img)

        self._variants = {
            **self._variants,
            "cropped": lambda i: i.center(size=180).square(),
            "red-only": lambda i: i.rgb_mute(green=True, blue=True),
            "blue-only": lambda i: i.rgb_mute(green=True, red=True),
            "green-only": lambda i: i.rgb_mute(blue=True, red=True),
            "red-blue-only": lambda i: i.rgb_mute(green=True),
            "red-green-only": lambda i: i.rgb_mute(blue=True),
        }

        self._actions = {
            **self._actions,
            "binary": lambda i, name: (
                i.gaussian(ksize=(7, 7))
                .threshold(
                    threshold=127.0,
                    threshold_type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU),
                )
                .write(append="{}_binary".format(name))
            ),
            "connected_components": lambda i, name: (
                i.gaussian(ksize=(7, 7))
                .threshold(
                    threshold=127.0,
                    threshold_type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU),
                )
                .connected_components(append=name, count=True)
            ),
        }


class Task4(Task):

    def __init__(self, img):
        super(Task4, self).__init__(img, 4)

        self._variants = {
            "method_1": lambda i: i,
            "method_2": lambda i: i.rgb_mute(blue=True, green=True),
            "method_3": lambda i: i.rgb_mute(blue=True, green=True).gaussian(
                ksize=(7, 7)
            ),
        }

        self._actions = {
            **self._actions,
            "kmeans_k_2": lambda i, name: i.kmeans(k=2, append="{}_k_2".format(name)),
            "kmeans_k_3": lambda i, name: i.kmeans(k=3, append="{}_k_3".format(name)),
        }


def runner(img_path, out_path, task):
    img = Image(img_path)
    with use_dir(out_path):
        task(img).run()


def main():
    tasks = [
        ("../data/diamond2.png", "results/task_1/diamond", Task1),
        ("../data/Dugong.jpg", "results/task_1/dugong", Task1Dugong),

        ("../data/diamond2.png", "results/task_2/diamond", Task2Diamond),
        ("../data/Dugong.jpg", "results/task_2/dugong", Task2Dugong),

        ("../data/diamond2.png", "results/task_3/diamond", DiamondTask3),
        ("../data/Dugong.jpg", "results/task_3/dugong", DugongTask3),

        ("../data/diamond2.png", "results/task_4/diamond", Task4),
        ("../data/Dugong.jpg", "results/task_4/dugong", Task4),
    ]

    p = Pool(cpu_count())
    p.starmap(runner, tasks)


if __name__ == "__main__":
    main()

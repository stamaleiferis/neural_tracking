import cv2
import numpy as np
import random


def draw_square(img, x, y, marker_size, xx, yy, theta):
    width, height = img.shape[0], img.shape[1]
    marker_size_large = marker_size * 2 ** 0.5

    lx_raw, rx_raw = x - marker_size, x + marker_size
    ly_raw, ry_raw = y - marker_size, y + marker_size

    lx, rx = x - marker_size_large, x + marker_size_large
    ly, ry = y - marker_size_large, y + marker_size_large

    lx, rx = np.clip(lx, 0, width), np.clip(rx, -1, width - 1)
    ly, ry = np.clip(ly, 0, height), np.clip(ry, -1, height - 1)

    lxi, lyi = int(lx), int(ly)
    rxi, ryi = int(np.ceil(rx)), int(np.ceil(ry))

    xx_r, yy_r = xx[lxi : rxi + 1, lyi : ryi + 1], yy[lxi : rxi + 1, lyi : ryi + 1]
    xx_r, yy_r = (
        np.cos(theta) * (xx_r - x) - np.sin(theta) * (yy_r - y) + x,
        np.sin(theta) * (xx_r - x) + np.cos(theta) * (yy_r - y) + y,
    )

    def intensity(val, left, right):
        return 1 - np.clip(np.maximum(left - val, val - right), 0, 1)

    darkness = 0.3 + 0.7 * np.random.random()
    scale = 1 - darkness * intensity(xx_r, lx_raw, rx_raw) * intensity(yy_r, ly_raw, ry_raw)
    for channel in range(3):
        img[lxi : rxi + 1, lyi : ryi + 1, channel] *= scale


def generate(xx, yy, img_blur=None, rng=0.0, W=48, H=48, N=6, M=6, degree=None):
    if img_blur is None:
        img_blur = (np.random.random((W // 3, H // 3, 3)) * 0.9) + 0.1
        img_blur = cv2.resize(img_blur, (H, W))

    yy_whole, xx_whole = np.meshgrid(np.arange(H), np.arange(W))

    img = img_blur + np.random.randn(W, H, 3) * 0.05 - 0.025

    for i in range(N):
        for j in range(M):
            r = yy[i, j]
            c = xx[i, j]

            if degree is None:
                theta = np.random.normal(0, 0.5) * 45 / 180 * np.pi
            else:
                theta = degree

            draw_square(img, r, c, 0.5 + rng * 1, xx_whole, yy_whole, theta)

    img[:, :1] *= np.random.random(img[:, :1].shape) * 0.5
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img, 0.0, 1.0)
    return img


def shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy):
    gaussian = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma ** 2))
    return xx + shear_x * gaussian, yy + shear_y * gaussian


def twist(center_x, center_y, sigma, theta, xx, yy):
    gaussian = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma ** 2))
    dx = xx - center_x
    dy = yy - center_y
    rotated_x = dx * np.cos(theta) - dy * np.sin(theta)
    rotated_y = dx * np.sin(theta) + dy * np.cos(theta)
    return xx + (rotated_x - dx) * gaussian, yy + (rotated_y - dy) * gaussian


def random_shear(xx, yy, W, H, interval=8):
    shear_ratio = 5
    center_x = random.random() * W
    center_y = random.random() * H
    sigma = random.random() * W / 2
    if np.random.random() < 0.3:
        normal = np.array([center_x - W / 2, center_y - H / 2])
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        shear_x = random.random() * interval * shear_ratio * normal[0]
        shear_y = random.random() * interval * shear_ratio * normal[1]
    else:
        shear_x = random.random() * interval * shear_ratio - interval * shear_ratio / 2
        shear_y = random.random() * interval * shear_ratio - interval * shear_ratio / 2
    return shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy)


def random_twist(xx, yy, W, H):
    twist_degree = 100
    center_x = random.random() * W
    center_y = random.random() * H
    sigma = random.random() * W / 2
    theta = (random.random() * twist_degree - twist_degree / 2.0) / 180.0 * np.pi
    return twist(center_x, center_y, sigma, theta, xx, yy)


def preprocessing(img, W, H):
    ret = img.copy()
    x_grid = np.arange(0, W, 1)
    y_grid = np.arange(0, H, 1)
    xx, yy = np.meshgrid(y_grid, x_grid)
    for _ in range(5):
        size_x = int(2 + random.random() * 15)
        size_y = int(2 + random.random() * 15)
        x = int(random.random() * (W - size_x))
        y = int(random.random() * (H - size_y))
        theta = np.random.random() * np.pi
        rng = 0.7
        xr = (xx - x) * np.cos(theta) - (yy - y) * np.sin(theta)
        yr = (xx - x) * np.sin(theta) + (yy - y) * np.cos(theta)
        mask = np.logical_and.reduce([(xr >= -size_x), (xr <= size_x), (yr >= -size_y), (yr <= size_y)])
        ret[mask] *= 1 + (np.random.random(3) * rng * 2 - rng)
    return ret


def generate_batch_fixed(batch_size=32, setting=(80, 112, 10, 14)):
    W, H, N, M = setting
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx0, yy0 = np.meshgrid(y, x)

    interval_x = W / N
    interval_y = H / M
    x_positions = np.arange(interval_x / 2, W, interval_x)[:N]
    y_positions = np.arange(interval_y / 2, H, interval_y)[:M]
    xind, yind = np.meshgrid(y_positions, x_positions)
    xind = xind.reshape([-1]).astype(np.int64)
    yind = yind.reshape([-1]).astype(np.int64)
    xind += (np.random.random(xind.shape) * 2 - 1).astype(np.int64)
    yind += (np.random.random(yind.shape) * 2 - 1).astype(np.int64)

    X, Y = [], []
    for _ in range(batch_size):
        xx = xx0 + (np.random.random(xx0.shape) * 2 - 1)
        yy = yy0 + (np.random.random(yy0.shape) * 2 - 1)
        rng = np.random.random()

        img0 = generate(
            xx[yind, xind].reshape([N, M]),
            yy[yind, xind].reshape([N, M]),
            img_blur=None,
            rng=rng,
            W=W,
            H=H,
            N=N,
            M=M,
            degree=0,
        )

        xx_distorted, yy_distorted = xx, yy
        xx_distorted, yy_distorted = random_shear(xx_distorted, yy_distorted, W, H)
        xx_distorted, yy_distorted = random_twist(xx_distorted, yy_distorted, W, H)
        xx_distorted += np.random.random(xx_distorted.shape) * 1 - 0.5
        yy_distorted += np.random.random(yy_distorted.shape) * 1 - 0.5

        img = generate(
            xx_distorted[yind, xind].reshape([N, M]),
            yy_distorted[yind, xind].reshape([N, M]),
            img_blur=None,
            rng=rng,
            W=W,
            H=H,
            N=N,
            M=M,
        )
        img = preprocessing(img, W, H)

        target = np.zeros([N, M, 2], dtype=np.float32)
        target[:, :, 0] = (
            xx_distorted[yind, xind].reshape([N, M]) - xx[yind, xind].reshape([N, M])
        )
        target[:, :, 1] = (
            yy_distorted[yind, xind].reshape([N, M]) - yy[yind, xind].reshape([N, M])
        )

        X.append(np.dstack([img0 - 0.5, img - 0.5]))
        Y.append(target)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y


def generate_batch_generic(batch_size=32, setting=None):
    X, Y = [], []
    if setting is None:
        N, M = np.random.randint(4, 15), np.random.randint(4, 15)
        W = np.random.randint(N * 6, 96)
        H = np.random.randint(M * 6, 96)
        W = (W // 16 + 1) * 16
        H = (H // 16 + 1) * 16
    else:
        W, H, N, M = setting

    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx, yy = np.meshgrid(y, x)

    interval_x = W / (N + 1)
    interval_y = H / (M + 1)
    x_positions = np.arange(interval_x, W, interval_x)[:N]
    y_positions = np.arange(interval_y, H, interval_y)[:M]
    xind, yind = np.meshgrid(x_positions, y_positions)
    xind = xind.reshape([-1]).astype(np.int64)
    yind = yind.reshape([-1]).astype(np.int64)
    xind += (np.random.random(xind.shape) * 4 - 2).astype(np.int64)
    yind += (np.random.random(yind.shape) * 4 - 2).astype(np.int64)

    for _ in range(batch_size):
        rng = np.random.random()
        img0 = generate(
            xx[xind, yind].reshape([N, M]),
            yy[xind, yind].reshape([N, M]),
            img_blur=None,
            rng=rng,
            W=W,
            H=H,
            N=N,
            M=M,
        )

        xx_distorted, yy_distorted = random_shear(xx, yy, W, H)
        xx_distorted, yy_distorted = random_twist(xx_distorted, yy_distorted, W, H)

        img = generate(
            xx_distorted[xind, yind].reshape([N, M]),
            yy_distorted[xind, yind].reshape([N, M]),
            img_blur=None,
            rng=rng,
            W=W,
            H=H,
            N=N,
            M=M,
        )

        target = np.zeros([W, H, 2], dtype=np.float32)
        target[:, :, 0] = xx_distorted - xx
        target[:, :, 1] = yy_distorted - yy

        features = np.dstack(
            [
                img0 - 0.5,
                img - 0.5,
                np.reshape(xx, [W, H, 1]),
                np.reshape(yy, [W, H, 1]),
            ]
        )
        X.append(features)
        Y.append(target)

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    # multi-scale downsampling of targets
    Y_list = [
        Y,
        Y[:, 1::2, 1::2],
        Y[:, 2::4, 2::4],
        Y[:, 4::8, 4::8],
        Y[:, 8::16, 8::16],
    ]
    return X, Y_list


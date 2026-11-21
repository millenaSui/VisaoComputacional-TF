"""
Módulo de funções utilitárias para processamento de imagens: ordenação de pontos,
cálculo de áreas, verificação de contenção, análise de textura, retificação de
perspectiva via pontos de fuga/distância focal adaptativa e detecção de pessoas.
"""

import cv2
import numpy as np


def sort_points(pts):
    """
    Ordena pontos na ordem: topo-esquerdo, topo-direito, base-direita, base-esquerda.

    :param pts: Array de pontos (4, 2).
    :return: Array de pontos ordenados (4, 2).
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_corners(contorno):
    """
    Seleciona os quatro cantos mais extremos de um contorno.

    :param contorno: Contorno aproximado (N, 1, 2).
    :return: Array de quatro pontos (4, 2).
    """
    pts = contorno.reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def calculate_area_quad(pts):
    """
    Calcula a área de um quadrilátero definido por quatro pontos (fórmula do polígono).

    :param pts: Array de pontos (4, 2).
    """
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def inner(inner_pts, outer_pts):
    """
    Verifica se o candidato está contido no alvo maior com margem elástica de 15%.

    :param inner_pts: Pontos do candidato (4, 2).
    :param outer_pts: Pontos do alvo maior (4, 2).
    :return: True se contido, False caso contrário.
    """
    ix, iy, iw, ih = cv2.boundingRect(inner_pts)
    ox, oy, ow, oh = cv2.boundingRect(outer_pts)

    margem_w = ow * 0.15
    margem_h = oh * 0.15

    return (
        ix >= ox - margem_w
        and iy >= oy - margem_h
        and ix + iw <= ox + ow + margem_w
        and iy + ih <= oy + oh + margem_h
    )


def calculate_inner_texture(gray_img, pts):
    """
    Mede a complexidade visual (textura/tinta) dentro de um quadrilátero.

    :param gray_img: Imagem em escala de cinza.
    :param pts: Pontos do quadrilátero (4, 2).
    :return: Variância da Laplaciana da região interna.
    """
    x, y, w, h = cv2.boundingRect(np.int32(pts))
    x, y = max(0, x), max(0, y)
    crop = gray_img[y:y + h, x:x + w]

    if crop.size == 0:
        return 0
    return cv2.Laplacian(crop, cv2.CV_64F).var()


def auto_canny(image, sigma=0.33):
    """
    Calcula os limites de borda dinamicamente com base na iluminação da imagem.

    :param image: Imagem em escala de cinza.
    :param sigma: Fator de ajuste para os limites.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def compute_vanishing_point_homography(pts_src, max_w, max_h):
    """
    Calcula a matriz de homografia (3x3) usando pontos de fuga e a linha do horizonte
    para remover a distorção projetiva, seguida de um alinhamento afim.

    :param pts_src: Pontos ordenados do quadro [tl, tr, br, bl] (4, 2).
    :param max_w: Largura da imagem de saída.
    :param max_h: Altura da imagem de saída.
    :return: Matriz de homografia 3x3 (float64).
    """
    pts_h = np.hstack([pts_src.astype(np.float64), np.ones((4, 1), dtype=np.float64)])
    tl, tr, br, bl = pts_h

    # Retas das bordas opostas
    l_top = np.cross(tl, tr)
    l_bottom = np.cross(bl, br)
    l_left = np.cross(tl, bl)
    l_right = np.cross(tr, br)

    # Pontos de fuga e linha do horizonte (linha no infinito)
    v1 = np.cross(l_top, l_bottom)
    v2 = np.cross(l_left, l_right)
    l_inf = np.cross(v1, v2)

    if np.abs(l_inf[2]) > 1e-12:
        l_inf = l_inf / l_inf[2]
        H_P = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [l_inf[0], l_inf[1], 1.0]
        ], dtype=np.float64)
    else:
        H_P = np.eye(3, dtype=np.float64)

    # Projeta os cantos originais com H_P (forma um paralelogramo)
    pts_affine_h = (H_P @ pts_h.T).T
    pts_affine = (pts_affine_h[:, :2] / pts_affine_h[:, 2:3]).astype(np.float32)

    # Alinhamento afim (3 pontos) para o retângulo final
    pts_dst = np.float32([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1]
    ])
    M_affine = cv2.getAffineTransform(pts_affine[:3], pts_dst)
    H_A = np.vstack([M_affine, [0.0, 0.0, 1.0]])

    return H_A @ H_P


def estimate_aspect_ratio_from_vanishing_points(pts_src, img_shape):
    """
    Estima a distância focal de forma adaptativa usando o ângulo de convergência
    das bordas opostas como peso de confiança, e calcula a proporção real 3D do quadro.

    :param pts_src: Pontos ordenados do quadro [tl, tr, br, bl] (4, 2).
    :param img_shape: Shape da imagem original (img.shape).
    :return: Razão largura/altura real (float) ou None se não for possível estimar.
    """
    h_img, w_img = img_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    max_dim = float(max(w_img, h_img))

    tl_2d, tr_2d, br_2d, bl_2d = pts_src.astype(np.float64)

    def angle_between(v_a, v_b):
        n_a, n_b = np.linalg.norm(v_a), np.linalg.norm(v_b)
        if n_a < 1e-8 or n_b < 1e-8:
            return 0.0
        cos_ang = np.clip(np.dot(v_a, v_b) / (n_a * n_b), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_ang)))

    angle_h = angle_between(tr_2d - tl_2d, br_2d - bl_2d)
    angle_v = angle_between(bl_2d - tl_2d, br_2d - tr_2d)
    min_angle = min(angle_h, angle_v)

    # Peso de confiança contínuo: 0.0 abaixo de 2°, cresce até 1.0 em 10°
    confidence = float(np.clip((min_angle - 2.0) / 8.0, 0.0, 1.0))

    pts_h = np.hstack([pts_src.astype(np.float64), np.ones((4, 1), dtype=np.float64)])
    tl, tr, br, bl = pts_h

    v1 = np.cross(np.cross(tl, tr), np.cross(bl, br))
    v2 = np.cross(np.cross(tl, bl), np.cross(tr, br))

    f_prior = 1.2 * max_dim
    f = f_prior

    if confidence > 0.0 and np.abs(v1[2]) > 1e-8 and np.abs(v2[2]) > 1e-8:
        v1_eucl = v1[:2] / v1[2]
        v2_eucl = v2[:2] / v2[2]
        f_sq = -((v1_eucl[0] - cx) * (v2_eucl[0] - cx) + (v1_eucl[1] - cy) * (v2_eucl[1] - cy))

        if f_sq > 0:
            f_est = float(np.sqrt(f_sq))
            f_est = float(np.clip(f_est, 0.7 * max_dim, 4.0 * max_dim))
            f = confidence * f_est + (1.0 - confidence) * f_prior

    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    normal = np.cross(K_inv @ v1, K_inv @ v2)
    if np.linalg.norm(normal) < 1e-12:
        return None

    pts_3d = []
    for p in pts_h:
        ray = K_inv @ p
        denom = np.dot(normal, ray)
        if np.abs(denom) < 1e-12:
            return None
        pts_3d.append(ray / denom)

    tl_3d, tr_3d, br_3d, bl_3d = pts_3d
    w_3d = 0.5 * (np.linalg.norm(tr_3d - tl_3d) + np.linalg.norm(br_3d - bl_3d))
    h_3d = 0.5 * (np.linalg.norm(bl_3d - tl_3d) + np.linalg.norm(br_3d - tr_3d))

    if h_3d < 1e-12:
        return None

    return w_3d / h_3d


def detect_people(img, model, conf=0.15, num_human_classes=6):
    """
    Detecta pessoas na imagem retificada usando YOLOE-26, ignorando
    classes distratoras (como 'painting' e 'landscape').

    :param img: Imagem retificada (BGR).
    :param model: Instância carregada do modelo YOLOE.
    :param conf: Limiar mínimo de confiança.
    :param num_human_classes: Número de classes humanas no início da lista do modelo.
    :return: Tupla (imagem_anotada, quantidade_de_pessoas).
    """
    results = model.predict(
        source=img,
        conf=conf,
        iou=0.45,
        imgsz=1024,
        agnostic_nms=True,
        verbose=False,
    )
    boxes = results[0].boxes

    img_annotated = img.copy()
    count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id >= num_human_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        count += 1

        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Pessoa {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(
            img_annotated, label, (x1 + 2, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    return img_annotated, count
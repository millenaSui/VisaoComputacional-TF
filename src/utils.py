"""
Módulo de funções utilitárias para processamento de imagens. Ele fornece 
funções para ordenar pontos, calcular áreas de quadriláteros, verificar
se um quadrilátero está contido dentro de outro e medir a complexidade
visual (textura) dentro de um quadrilátero.
"""

import cv2
import numpy as np

def sort_points(pts):
    """
    Ordena pontos colocando-os em ordem: topo-esquerdo, topo-direito, 
    base-direita, base-esquerda, para garantir consistência na retificação
    da perspectiva.

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
    Seleciona os quatro cantos de um contorno aproximado, visando
    encontrar os pontos mais extremos de cada lado do contorno.

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
    Calcula a área de um quadrilátero definido por quatro pontos
    usando a fórmula do polígono.

    :param pts: Array de pontos (4, 2).
    """

    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def inner(inner_pts, outer_pts):
    """
    Verifica se o candidato está fisicamente contido no alvo maior.
    Usa margens elásticas de 15% para suportar perspectivas distorcidas.
    
    :param inner_pts: Pontos do candidato (4, 2).
    :param outer_pts: Pontos do alvo maior (4, 2).

    :return: True se o candidato estiver dentro do alvo, False caso contrário.
    """
    ix, iy, iw, ih = cv2.boundingRect(inner_pts)
    ox, oy, ow, oh = cv2.boundingRect(outer_pts)

    margem_w = ow * 0.15
    margem_h = oh * 0.15

    reslt = (ix >= ox - margem_w and iy >= oy - margem_h and
             ix + iw <= ox + ow + margem_w and iy + ih <= oy + oh + margem_h)

    return reslt


def calculate_inner_texture(gray_img, pts):
    """
    Mede a complexidade visual (textura/tinta) dentro de um quadrilátero.
    
    :param gray_img: Imagem em escala de cinza.
    :param pts: Pontos do quadrilátero (4, 2).

    :return: Variância da Laplaciana da região interna.
    """
    x, y, w, h = cv2.boundingRect(np.int32(pts))
    x, y = max(0, x), max(0, y)
    crop = gray_img[y:y+h, x:x+w]

    if crop.size == 0:
        return 0
    return cv2.Laplacian(crop, cv2.CV_64F).var()


def auto_canny(image, sigma=0.33):
    """
    Calcula os limites das border dinamicamente com base na iluminação da própria imagem.
    
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

    # equações das retas das bordas opostas
    l_top = np.cross(tl, tr)
    l_bottom = np.cross(bl, br)
    l_left = np.cross(tl, bl)
    l_right = np.cross(tr, br)

    # pontos de fuga
    v1 = np.cross(l_top, l_bottom)
    v2 = np.cross(l_left, l_right)

    l_inf = np.cross(v1, v2) # linha do horizonte

    if np.abs(l_inf[2]) > 1e-12:
        l_inf = l_inf / l_inf[2]
        H_P = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [l_inf[0], l_inf[1], 1.0]
        ], dtype=np.float64)
    else:
        H_P = np.eye(3, dtype=np.float64)

    pts_affine_h = (H_P @ pts_h.T).T
    pts_affine = (pts_affine_h[:, :2] / pts_affine_h[:, 2:3]).astype(np.float32)

    # transformação afim
    pts_dst = np.float32([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1]
    ])
    M_affine = cv2.getAffineTransform(pts_affine[:3], pts_dst)

    H_A = np.vstack([M_affine, [0.0, 0.0, 1.0]])

    H_total = H_A @ H_P
    return H_total

def estimate_aspect_ratio_from_vanishing_points(pts_src, img_shape):
    """
    Estima a distância focal da câmera a partir da ortogonalidade dos pontos de fuga
    e calcula a proporção real (largura / altura) do quadro no espaço 3D.

    :param pts_src: Pontos ordenados do quadro [tl, tr, br, bl] (4, 2).
    :param img_shape: Shape da imagem original (img.shape).
    :return: Razão largura/altura real (float) ou None se não for possível estimar.
    """
    h_img, w_img = img_shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    max_dim = max(w_img, h_img)

    pts_h = np.hstack([pts_src.astype(np.float64), np.ones((4, 1), dtype=np.float64)])
    tl, tr, br, bl = pts_h

    # retas das bordas opostas
    l_top = np.cross(tl, tr)
    l_bottom = np.cross(bl, br)
    l_left = np.cross(tl, bl)
    l_right = np.cross(tr, br)

    # pontos de fuga (horizontal e vertical)
    v1 = np.cross(l_top, l_bottom)
    v2 = np.cross(l_left, l_right)

    # estima a distância focal f
    f = float(max_dim)
    if np.abs(v1[2]) > 1e-8 and np.abs(v2[2]) > 1e-8:
        v1_eucl = v1[:2] / v1[2]
        v2_eucl = v2[:2] / v2[2]
        f_sq = -((v1_eucl[0] - cx) * (v2_eucl[0] - cx) + (v1_eucl[1] - cy) * (v2_eucl[1] - cy))
        if f_sq > 0:
            f_est = np.sqrt(f_sq)
            # garante um valor de distância focal plausível contra ruído nos cantos
            if 0.3 * max_dim < f_est < 5.0 * max_dim:
                f = f_est

    # matriz intrínseca K e sua inversa
    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # direções 3D dos pontos de fuga e vetor normal do plano do quadro
    d1 = K_inv @ v1
    d2 = K_inv @ v2
    normal = np.cross(d1, d2)

    if np.linalg.norm(normal) < 1e-12:
        return None

    # interseção dos raios dos cantos com o plano
    pts_3d = []
    for p in pts_h:
        ray = K_inv @ p
        denom = np.dot(normal, ray)
        if np.abs(denom) < 1e-12:
            return None
        pts_3d.append(ray / denom)

    tl_3d, tr_3d, br_3d, bl_3d = pts_3d

    # largura e altura reais no espaço 3D
    w_3d = 0.5 * (np.linalg.norm(tr_3d - tl_3d) + np.linalg.norm(br_3d - bl_3d))
    h_3d = 0.5 * (np.linalg.norm(bl_3d - tl_3d) + np.linalg.norm(br_3d - tr_3d))

    if h_3d < 1e-12:
        return None

    return w_3d / h_3d
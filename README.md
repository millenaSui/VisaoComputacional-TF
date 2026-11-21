# TA03 - Visão Computacional (Transformação de Perspectiva) 

A aplicação foi projetada para receber um conjunto de fotografias contendo quadros ou obras de arte emolduradas, processar essas imagens para identificar a região exata da tela/pintura e aplicar uma transformação de perspectiva (homografia). O resultado é uma imagem plana e retificada da obra, corrigindo o ângulo em que a foto foi tirada e isolando a arte de sua moldura e do ambiente externo.

## Fluxo de Trabalho

O fluxo de execução da aplicação é automatizado e dividido nas seguintes etapas:

1. **Leitura e Pré-Processamento:** As fotografias, que devem estar presentes no diretório `./dataset`, são carregadas, redimensionadas proporcionalmente para otimizar o processamento e convertidas para escala de cinza (*grayscale*) com aplicação de desfoque Gaussiano para redução de ruído.
2. **Detecção de Características e Bordas:** O algoritmo utiliza múltiplas abordagens combinadas para mapear bordas na imagem: 
   - Detecção de bordas adaptativa (`auto_canny`).
   - Gradientes morfológicos para destacar relevos de molduras complexas.
   - Limiarização adaptativa para lidar com variações locais de iluminação.
3. **Seleção do Melhor Quadro:** A partir dos contornos encontrados, a aplicação busca polígonos de quatro lados (quadriláteros). Para garantir que a tela da pintura seja selecionada (e não a moldura externa), o sistema avalia a área, verifica se há quadros contidos uns nos outros e calcula a complexidade visual/textura da região interna usando a variância da Laplaciana.
4. **Retificação de Perspectiva:** Calculam-se os pontos de fuga a partir da interseção das bordas opostas e determina-se a linha do horizonte para remover a distorção projetiva. Em seguida, estima-se a distância focal da câmera para recuperar a proporção real (altura/largura) do quadro no espaço 3D e compõe-se a retificação projetiva com um alinhamento afim (`cv2.getAffineTransform`). Por fim, aplica-se a matriz de homografia resultante (`cv2.warpPerspective`), tornando a imagem retangular, plana e fiel às dimensões originais da obra.
5. **Corte e Salvamento:** Uma pequena margem de segurança (3.5%) é recortada das bordas para eliminar eventuais sombras ou reflexos residuais da moldura física. A imagem final tratada é salva no diretório `./output`.

6. **Detecção humana:** o YOLOE recebe somente classes humanas (person, human, woman, man, child e baby). A imagem é avaliada em sua versão original e com contraste CLAHE; detecções com baixa confiança só são aceitas quando confirmadas pelas duas versões. Filtros de área e proporção reduzem objetos que não tenham forma compatível com uma pessoa.

## Execução

Devido às dependências de sistema exigidas pelo `OpenCV`, a aplicação foi conteinerizada via **Docker** utilizando a versão `opencv-python-headless`. Isso garante um ambiente de execução leve e isolado, eliminando a necessidade de configurar dependências de interface gráfica (X11) ou instalar pacotes na máquina *host*.

### Pré-requisitos
Certifique-se de ter o [Docker](https://docs.docker.com/get-docker/) instalado no seu sistema. Coloque as fotografias que deseja processar (formatos suportados: `.jpg`, `.jpeg`, `.png`, `.bmp`) na pasta `./dataset/`.

### Usando Docker Compose (Recomendado)
A maneira mais fácil de executar o projeto é utilizando o Docker Compose, que constrói a imagem e mapeia automaticamente os volumes dos diretórios de entrada e saída.

No terminal, na raiz do projeto, execute:

```bash
docker-compose up --build
```

O Docker constrói as dependências uma vez e reutiliza o cache de camadas nas
execuções seguintes. `requirements.txt` fixa as versões principais e volumes
separam os diretórios de entrada, saída e modelo do sistema da imagem.
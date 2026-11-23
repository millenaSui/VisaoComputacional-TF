# TF - Visão Computacional (Reconhecimento de Pessoas em Obras Retificadas)

Esta aplicação representa a evolução do **TA03** da disciplina de Visão Computacional. Partindo do *pipeline* original de detecção de obras de arte emolduradas e correção de perspectiva (homografia), o projeto foi expandido para incorporar o **reconhecimento de pessoas** nas obras retificadas, utilizando o modelo **YOLOE-26** (`ultralytics`).

A aplicação recebe um conjunto de fotografias contendo quadros ou pinturas emolduradas, isola a região exata da tela, corrige a distorção de perspectiva preservando a proporção real da obra e, por fim, identifica e anota as figuras humanas presentes na pintura.

## Fluxo de Trabalho

O pipeline de execução é totalmente automatizado e dividido nas seguintes etapas:

1. **Leitura e Pré-Processamento:** As fotografias presentes no diretório `./dataset` são carregadas, redimensionadas proporcionalmente (altura fixa de 800 px para a etapa de detecção) e convertidas para escala de cinza (*grayscale*) com aplicação de desfoque Gaussiano (`7x7`) para redução de ruído.
2. **Detecção de Características e Bordas:** O algoritmo combina três estratégias complementares para mapear as bordas da imagem:
   - Detecção de bordas adaptativa baseada na mediana de iluminação (`auto_canny`).
   - Gradiente morfológico para evidenciar relevos físicos de molduras complexas.
   - Limiarização adaptativa Gaussiana para lidar com variações locais de iluminação.
3. **Seleção do Melhor Quadro:** A partir dos contornos extraídos, o sistema filtra regiões sólidas (solidez $\ge 0{,}85$) e busca quadriláteros candidatos, eliminando duplicatas próximas. Para distinguir a pintura da moldura externa, verifica-se a contenção entre retângulos candidatos e avalia-se a complexidade visual interna por meio da variância da Laplaciana.
4. **Retificação de Perspectiva e Proporção 3D:**
   - Calculam-se os pontos de fuga a partir da interseção das bordas opostas e determina-se a linha do horizonte para remover a distorção projetiva.
   - A distância focal da câmera é estimada de forma adaptativa — ponderando a estimativa geométrica pelo ângulo de convergência das bordas — para recuperar a proporção real 3D (largura/altura) da obra.
   - A transformação projetiva é composta com um alinhamento afim (`cv2.getAffineTransform`) e aplicada à imagem original via `cv2.warpPerspective`.
5. **Recorte de Margens:** Uma margem de segurança é recortada de cada borda da imagem retificada para eliminar eventuais sobras de moldura, sombras ou reflexos residuais.
6. **Reconhecimento de Elementos com YOLOE-26:** Sobre a obra já retificada e recortada, executa-se o modelo **YOLOE-26** (`yoloe-26l-seg.pt`) configurado com vocabulário aberto:
   - **Classes humanas (alvo):** `person`, `woman`, `man`, `child`, `baby`, `human`.
   - **Classes distratoras (descartadas na pós-filtragem):** `painting`, `artwork`, `picture frame`, `landscape`, `nature` — incluídas para reduzir falsos positivos causados pelo próprio estilo pictórico da obra.
   - A inferência utiliza `conf=0.15`, `iou=0.45`, `imgsz=1024` e `agnostic_nms=True`. Cada pessoa identificada recebe uma caixa delimitadora (*bounding box*) com o rótulo `Pessoa` e seu respectivo índice de confiança, além de ter o total exibido no terminal.
7. **Salvamento:** A imagem final retificada e anotada é salva no diretório `./output` mantendo o nome do arquivo original.

## Estrutura do Projeto

- `main.py`: Ponto de entrada da aplicação; lista as imagens de entrada, carrega o modelo YOLOE-26 com as classes alvo e distratoras e dispara o processamento em lote.
- `src/images.py`: Responsável pela localização do melhor quadro (`find_best_frame`) e pelo pipeline completo de retificação, recorte, detecção de pessoas e salvamento (`process_images`).
- `src/utils.py`: Funções utilitárias de geometria, análise de textura, cálculo de homografia por pontos de fuga, estimativa adaptativa de proporção 3D e inferência/anotação com YOLOE (`detect_people`).
- `dataset/`: Diretório de entrada para as fotografias das obras.
- `output/`: Diretório onde as imagens retificadas e anotadas são salvas.

## Execução

Devido às dependências de sistema exigidas pelo `OpenCV` e pelo ambiente de inferência do `YOLOE`, a aplicação foi conteinerizada via **Docker** utilizando `opencv-python-headless`. Isso garante um ambiente leve, reprodutível e isolado, eliminando a necessidade de configurar interface gráfica (X11) ou instalar bibliotecas diretamente na máquina *host*.

### Pré-requisitos

Certifique-se de ter o [Docker](https://docs.docker.com/get-docker/) instalado no seu sistema. Coloque as fotografias que deseja processar (formatos suportados: `.jpg`, `.jpeg`, `.png`, `.bmp`) na pasta `./dataset/`.

### Usando Docker Compose (Recomendado)

A maneira mais simples de executar o projeto é utilizando o Docker Compose, que constrói a imagem e mapeia automaticamente os volumes dos diretórios de entrada, saída e pesos do modelo.

No terminal, na raiz do projeto, execute:

```bash
docker-compose up --build
```
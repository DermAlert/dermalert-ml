# Pre-classificação de Risco de Câncer de Pele

Este projeto tem como objetivo desenvolver, testar e otimizar um modelo de aprendizado de máquina para pré-classificação de risco de câncer de pele usando imagens dermatoscópicas da base de dados ISIC Archive.

---

## Sumário

- [Visão Geral](#visão-geral)
- [Base de Dados](#base-de-dados)
- [Fluxo de Trabalho](#fluxo-de-trabalho)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Aquisição e Exploração dos Dados](#aquisição-e-exploração-dos-dados)
- [Preparação dos Dados](#preparação-dos-dados)
- [Teste e Seleção de Modelos](#teste-e-seleção-de-modelos)
- [Otimização e Ajustes Finos](#otimização-e-ajustes-finos)
- [Avaliação](#avaliação)
- [Como Contribuir](#como-contribuir)
- [Licença](#licença)
- [Contato](#contato)

---

## Visão Geral

O objetivo deste repositório é criar um pipeline completo de pré-classificação de risco de câncer de pele, desde a aquisição e exploração dos dados até a otimização do modelo. O foco principal é o uso de técnicas de visão computacional e aprendizado profundo para auxiliar na triagem de lesões cutâneas.

## Base de Dados

Este projeto utiliza a base de dados disponível em:

- **ISIC Archive**: [https://www.isic-archive.com/](https://www.isic-archive.com/)

A base contém imagens dermatoscópicas rotuladas com diferentes tipos de lesões, incluindo melanoma e nevos benignos.

## Fluxo de Trabalho

1. **Aquisição e Exploração dos Dados**: download, inspeção e análise exploratória das imagens e metadados.
2. **Preparação dos Dados**: pré-processamento, balanceamento, aumento de dados (data augmentation).
3. **Teste e Seleção de Modelos**: definição de arquiteturas candidatas (por exemplo, CNNs, transfer learning), experimentação e comparação de desempenho.
4. **Otimização e Ajustes Finos**: ajuste de hiperparâmetros, fine-tuning de modelos pré-treinados, validação cruzada e análise de métricas.

## Estrutura do Projeto

```
├── data/                  # Dados brutos e processados
├── notebooks/             # Jupyter notebooks de exploração e experimentos
├── src/                   # Código-fonte do pipeline
│   ├── data/              # Scripts de aquisição e pré-processamento
│   ├── models/            # Definições de modelos e treinamento
│   └── utils/             # Funções auxiliares
├── models/                # Modelos treinados e checkpoints
├── reports/               # Relatórios, gráficos e resultados
├── requirements.txt       # Dependências do Python
└── README.md              # Documentação deste repositório
```

## Requisitos

- Python 3.13
- uv (gerenciador de pacotes e ambientes) – instale seguindo [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- Bibliotecas de Python (definidas em `requirements.txt`):
  - numpy, pandas, scikit-learn
  - matplotlib, seaborn
  - tensorflow ou pytorch
  - albumentations (opcional para data augmentation)
  - opencv-python

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repo-cancer-pele.git
   cd seu-repo-cancer-pele
   ```
2. Instale o uv (caso ainda não tenha):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Instale o Python 3.13 via uv:
   ```bash
   uv python install 3.13
   ```
4. Crie e ative o ambiente virtual:
   ```bash
   uv venv
   source .venv/bin/activate  # Linux/macOS
   ```
5. Instale as dependências do projeto com uv:
   ```bash
   uv pip sync requirements.txt
   ```

## Aquisição e Exploração dos Dados

1. Autentique-se no ISIC Archive e obtenha as credenciais de acesso.
2. Utilize o script em `src/data/download_isic.py` para baixar imagens e metadados.
3. Execute o notebook `notebooks/exploracao_dados.ipynb` para visualizar distribuições de classes e exemplos de imagens.

## Preparação dos Dados

- Conversão de formatos de imagem e normalização.
- Balanceamento de classes (undersampling/oversampling).
- Aumento de dados (rotations, flips, ajustes de brilho/contraste).
- Divisão em conjuntos de treino, validação e teste.

Scripts: `src/data/preprocess.py`

## Teste e Seleção de Modelos

- Definição de arquiteturas (CNN do zero ou transfer learning, ex.: ResNet, EfficientNet).
- Treinamento inicial para comparação de acurácia, sensibilidade e especificidade.

Notebooks: `notebooks/teste_modelos.ipynb`

## Otimização e Ajustes Finos

- Grid Search ou Random Search para hiperparâmetros.
- Fine-tuning de modelos pré-treinados com camadas adicionais.
- Validação cruzada e análise de curvas de aprendizagem.

Notebooks: `notebooks/otimizacao_hiperparametros.ipynb`

## Avaliação

- Métricas: AUC-ROC, precisão, recall e F1-score.
- Matriz de confusão e gráficos de curvas ROC.
- Relatório de resultados em `reports/`.

## Como Contribuir

1. Fork este repositório.
2. Crie uma branch: `git checkout -b minha-nova-feature`.
3. Faça commits das suas alterações: `git commit -m 'Adiciona feature X'`.
4. Envie para o branch original: `git push origin minha-nova-feature`.
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contato

- **Autor:** Giovanni Giampauli

- **E-mail:** giovanni.acg@gmail.com

- **Repositório:** https://github.com/DermAlert/dermalert-ml

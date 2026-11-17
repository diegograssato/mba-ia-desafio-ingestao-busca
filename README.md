# Ingestão e Busca Semântica com LangChain e Postgres

## Objetivo

Você deve entregar um software capaz de:

- Ingestão: Ler um arquivo PDF e salvar suas informações em um banco de dados PostgreSQL com extensão pgVector.
- Busca: Permitir que o usuário faça perguntas via linha de comando (CLI) e receba respostas baseadas apenas no conteúdo do PDF.

## Exemplo no CLI

Faça sua pergunta:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

Perguntas fora do contexto:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Tecnologias obrigatórias

- Linguagem: Python
- Framework: LangChain
- Banco de dados: PostgreSQL + pgVector
- Execução do banco de dados: Docker & Docker Compose (docker-compose fornecido no repositório de exemplo)

## Pacotes recomendados

- Split: `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- Embeddings (OpenAI): `from langchain_openai import OpenAIEmbeddings`
- Embeddings (Gemini): `from langchain_google_genai import GoogleGenerativeAIEmbeddings`
- PDF: `from langchain_community.document_loaders import PyPDFLoader`
- Ingestão: `from langchain_postgres import PGVector`
- Busca: `similarity_search_with_score(query, k=10)`

## OpenAI

- Crie uma API Key da OpenAI.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial da OpenAI para ver os modelos disponíveis.

## Uso da Azure

Esta implementação usa o **Azure OpenAI / Azure AI Foundry** para duas tarefas:

- gerar os embeddings dos chunks durante a ingestão;
- gerar a resposta final do chat a partir dos chunks recuperados.

O PostgreSQL com pgVector não é provisionado na Azure neste laboratório: ele é executado localmente pelo `docker-compose.yml`. O fluxo é:

```text
document.pdf -> PyPDFLoader -> chunks -> Azure OpenAI (embeddings)
             -> PostgreSQL/pgVector -> busca por similaridade
             -> Azure OpenAI (chat) -> resposta no terminal
```

### Recursos e deployments

No Azure AI Foundry ou no recurso Azure OpenAI, crie dois deployments:

1. um deployment de modelo de chat, como o configurado em `AZURE_OPENAI_DEPLOYMENT`;
2. um deployment de embeddings, como o configurado em `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME`.

O nome do deployment é definido por você na Azure e é esse nome que a API recebe no campo `model`. Ele pode ser diferente do nome do modelo-base.

### Configuração local

Copie o template de configuração e preencha o arquivo `.env` sem versionar a chave:

```bash
cp .env.example .env
```

Variáveis utilizadas pela integração Azure:

| Variável | Finalidade | Exemplo de formato |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Chave de acesso ao recurso Azure OpenAI | valor secreto obtido na Azure |
| `AZURE_OPENAI_BASE_URL` | Endpoint v1 usado pelo modelo de chat | `https://<recurso>.openai.azure.com/openai/v1/` |
| `AZURE_OPENAI_DEPLOYMENT` | Nome do deployment de chat | `gpt-4.1` |
| `AZURE_OPENAI_EMBEDDINGS_BASE_URL` | Endpoint v1 usado para embeddings | `https://<recurso>.openai.azure.com/openai/v1/` |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_MODEL` | Modelo informado ao wrapper de embeddings | `text-embedding-3-small` |
| `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME` | Nome exato do deployment de embeddings na Azure | `text-embedding-3-small` |

Se chat e embeddings estiverem no mesmo recurso, os dois `BASE_URL` podem ter o mesmo valor. Mantenha a barra final e o sufixo `/openai/v1/`.

As variáveis `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_MODEL` e `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT` permanecem no template por compatibilidade, mas não são lidas pelo código atual.

> Segurança: nunca envie o `.env` ou a API key ao repositório. Em produção, prefira autenticação com Microsoft Entra ID e um cofre de segredos, além de limites de custo e monitoramento no recurso Azure.

Referência: [documentação da API v1 do Azure OpenAI](https://learn.microsoft.com/azure/ai-foundry/openai/api-version-lifecycle).

## Gemini

- Crie uma API Key da Google.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial do Google para ver os modelos disponíveis.

Os limites de requisições gratuitas dos modelos podem mudar com frequência. Para informações atualizadas, consulte a documentação oficial do Google.

## Escolha dos modelos

Este desafio não fixa modelos. Nomes e versões mudam com frequência e alguns são descontinuados, então faz parte do desafio consultar a documentação oficial do provedor que você escolher, ver quais modelos estão disponíveis no momento e selecionar os que atendem ao objetivo. Para o volume deste desafio, os modelos mais leves e baratos de cada provedor são suficientes.

Atenção: modelos de embedding diferentes geram vetores com dimensões diferentes. A tabela de vetores é criada na primeira ingestão, já com a dimensão do modelo que você escolheu. Se você trocar de modelo de embeddings depois disso, a ingestão passa a falhar por incompatibilidade de dimensão. Nesse caso é responsabilidade sua apagar a collection existente (ou o volume do banco) e refazer a ingestão do zero com o novo modelo.

## Requisitos

### 1. Ingestão do PDF

- O PDF deve ser dividido em chunks de 1000 caracteres com overlap de 150.
- Cada chunk deve ser convertido em embedding.
- Os vetores devem ser armazenados no banco de dados PostgreSQL com pgVector.

### 2. Consulta via CLI

Criar um script Python para simular um chat no terminal.

Passos ao receber uma pergunta:

- Vetorizar a pergunta.
- Buscar os 10 resultados mais relevantes (k=10) no banco vetorial.
- Montar o prompt e chamar a LLM.
- Retornar a resposta ao usuário.

Prompt a ser utilizado:

```
CONTEXTO:
{resultados concatenados do banco de dados}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

## Estrutura obrigatória do projeto

Faça um fork do repositório para utilizar a estrutura abaixo: https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```

## VirtualEnv para Python

Crie e ative um ambiente virtual antes de instalar dependências:

```
python3 -m venv venv
source venv/bin/activate
```

## Ordem de execução

0. Instalar as dependências e configurar o ambiente:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Preencha no `.env` a conexão do PostgreSQL, o caminho do PDF e as variáveis Azure descritas acima.

1. Subir o banco de dados:

```
docker compose up -d
```

2. Executar ingestão do PDF:

```
python src/ingest.py
```

3. Rodar o chat:

```
python src/chat.py
```

## Entregável

Repositório público no GitHub contendo todo o código-fonte e README com instruções claras de execução do projeto.

## Referência complementar

Como material complementar sobre LangChain e os conceitos utilizados neste projeto, consulte o curso que ministrei:

- [LangChain for Beginners](https://github.com/diegograssato/langchain-for-beginners)

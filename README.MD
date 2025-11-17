# Chat Multi Usuários

Aplicação de chat multiusuário construída em Node.js com Express, LangChain e OpenAI, permitindo múltiplas sessões independentes de conversa com histórico, temperatura e *system prompt* configuráveis por sessão. 

---

## ✨ Funcionalidades

- ✅ **Sessões de chat multiusuário** via `sessionId` (UUID)
- 💾 **Histórico de mensagens por sessão** usando `LangChain` (`ChatMessageHistory`)
- 🧠 **Configuração dinâmica por sessão**:
  - `systemPrompt` (instruções do sistema por sessão)
  - `temperature` (0–1 na API; controle via slider no front-end)
- ⏱️ **Limpeza automática de sessões inativas** (30 minutos sem atividade)
- 🌐 **API REST documentada com Swagger** em `/api-docs`
- 💻 **Interface web pronta**:
  - Tela inicial com “Nova Sessão” e “Continuar Sessão”
  - Campo para colar o `sessionId` e retomar conversas
  - Slider de temperatura e campo de *system prompt*
  - Botões de “Limpar Histórico”, “Ver Histórico” e “Nova Sessão”
  - Botão para **copiar ID da sessão** (clipboard)

---

## 🧱 Tecnologias Utilizadas

- **Node.js** + **Express**
- **LangChain** (`@langchain/openai`, `@langchain/core`, `langchain`)
- **OpenAI API** (modelo padrão: `gpt-4.1-nano`)
- **Swagger UI** + `swagger.yaml` para documentação da API
- **CORS**, `dotenv`, `uuid`, `yamljs`
- Front-end em **HTML + CSS + JavaScript puro**

---

## 🏗️ Arquitetura Geral

### Backend (Node + Express)

- Arquivo principal: `server.js`
- Armazena sessões em memória usando um `Map`:

  - Cada sessão contém:
    - `messageHistory` (`ChatMessageHistory`)
    - `systemPrompt`
    - `temperature`
    - `createdAt`
    - `lastActivity`

- Uso de `RunnableWithMessageHistory` para integrar LangChain + histórico por sessão.

#### Principais rotas da API

- `POST /api/session/new`  
  Cria uma nova sessão e retorna um `sessionId`.

- `GET /api/session/:sessionId/exists`  
  Verifica se uma sessão existe e retorna informações básicas.

- `POST /api/chat`  
  Envia uma mensagem para a sessão:
  ```json
  {
    "sessionId": "uuid-da-sessao",
    "message": "Olá!",
    "temperature": 0.7,
    "systemPrompt": "Você é um assistente educado."
  }
````

* `GET /api/session/:sessionId/history`
  Retorna o histórico de mensagens da sessão.

* `DELETE /api/session/:sessionId/clear`
  Limpa o histórico de uma sessão.

* `PUT /api/session/:sessionId/config`
  Atualiza `systemPrompt` e/ou `temperature` da sessão.

* `DELETE /api/session/:sessionId`
  Remove completamente a sessão.

* `GET /api/status`
  Retorna status básico do servidor (uptime, número de sessões).

* `GET /api-docs`
  Interface do Swagger UI, carregando `swagger.yaml`.

> ⚠️ As sessões são mantidas **em memória** e não persistem se o servidor for reiniciado. criadas para fins de demonstração.

---

### Frontend (Interface Web)

* Arquivo principal: `index.html` 
* Single Page App simples, consumindo a API via `fetch`.

Principais características:

* Tela de boas-vindas com duas opções:

  * **➕ Nova Sessão** → faz `POST /api/session/new` e inicia o chat
  * **🔄 Continuar Sessão** → permite colar um `sessionId` e verifica em `/api/session/:sessionId/exists`
* Área de chat com:

  * Balões de mensagem para usuário e bot
  * Indicador de “digitando” animado
* Painel de parâmetros:

  * Slider de **temperatura** (`input range`)
  * `textarea` para **system prompt** da sessão
* Botões de ação:

  * **Limpar Histórico** → `DELETE /session/:id/clear`
  * **Ver Histórico** → `GET /session/:id/history` (mostra resumo em `alert` + `console.log`)
  * **Nova Sessão** → recarrega a página
* Botão **“Copiar ID da sessão”** usando `navigator.clipboard` ou *fallback* com `document.execCommand('copy')`.

> 🔧 No topo do script existe a constante `API_URL`. Ajuste para apontar para a URL/porta onde o backend está rodando (por exemplo, `http://localhost:3000/api`).

---

## 📦 Requisitos

* **Node.js** (recomendado >= 18)
* **npm** ou **yarn**
* Chave da API da OpenAI (`OPENAI_API_KEY`)

---

## ⚙️ Instalação

1. Clone o repositório ou copie os arquivos do projeto:

   ```bash
   git clone [url-do-repositorio](https://github.com/DevUnusual/avaliacao_comp_aplicada.git)
   cd avaliacao_comp_aplicada
   ```

2. Instale as dependências:

   ```bash
   npm install
   ```

3. Crie um arquivo `.env` na raiz do projeto:

   ```env
   OPENAI_API_KEY=coloque_sua_chave_aqui
   PORT=3000            # opcional, padrão 3000
   ```

---

## ▶️ Execução

1. Inicie o servidor:

   ```bash
   node server.js
   ```

2. Acompanhe o log no terminal; serão mostradas URLs como:

   * Interface (front-end): `http://localhost:3000`
   * Status: `http://localhost:3000/api/status`
   * Swagger UI: `http://localhost:3000/api-docs`

3. Abra o navegador e acesse a interface do chat:

   ```text
   http://localhost:3000
   ```

4. Escolha:

   * **Nova Sessão** para começar do zero
   * **Continuar Sessão** se já tiver um `sessionId` salvo

---

## 🧪 Exemplos de Uso com API (via cURL)

### Criar nova sessão

```bash
curl -X POST http://localhost:3000/api/session/new
```

### Enviar mensagem

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "SEU_SESSION_ID",
    "message": "Olá, tudo bem?",
    "temperature": 0.7,
    "systemPrompt": "Você é um assistente divertido."
  }'
```

### Ver histórico da sessão

```bash
curl http://localhost:3000/api/session/SEU_SESSION_ID/history
```

---
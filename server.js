require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');
const { ChatOpenAI } = require('@langchain/openai');
const { ChatPromptTemplate, MessagesPlaceholder } = require('@langchain/core/prompts');
const { RunnableWithMessageHistory } = require('@langchain/core/runnables');
const { ChatMessageHistory } = require('langchain/stores/message/in_memory');
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3333;
const swaggerDocument = YAML.load(path.join(__dirname, 'swagger.yaml'));

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// Armazenamento de sessões em memória
const sessions = new Map();

// Configurar modelo OpenAI padrão
const model = new ChatOpenAI({
  modelName: 'gpt-4.1-nano',
  temperature: 0.5,
  openAIApiKey: process.env.OPENAI_API_KEY,
  maxTokens: 2000,
});

// Template do prompt
const prompt = ChatPromptTemplate.fromMessages([
  ['system', '{system_prompt}'],
  new MessagesPlaceholder('chat_history'),
  ['human', '{input}'],
]);

// Criar chain com histórico
const chain = prompt.pipe(model);


app.get('/', (req, res) => {
  res.sendFile(__dirname + '/public/index.html');
});

//funcao para o historico
app.get('/api/session/:sessionId/exists', (req, res) => {
  const { sessionId } = req.params;
  
  if (sessions.has(sessionId)) {
    const session = sessions.get(sessionId);
    res.json({
      exists: true,
      sessionId,
      createdAt: session.createdAt,
      lastActivity: session.lastActivity,
      messageCount: session.messageHistory.messages?.length || 0,
    });
  } else {
    res.json({
      exists: false,
      sessionId,
    });
  }
});

const getMessageHistory = (sessionId) => {
  if (!sessions.has(sessionId)) {
    sessions.set(sessionId, {
      messageHistory: new ChatMessageHistory(),
      systemPrompt: 'Você é um assistente útil e prestativo.',
      temperature: 0.5,
      createdAt: new Date(),
      lastActivity: new Date(),
    });
  } else {
    // Atualizar última atividade
    sessions.get(sessionId).lastActivity = new Date();
  }
  return sessions.get(sessionId).messageHistory;
};

// Chain com histórico de mensagens
const chainWithHistory = new RunnableWithMessageHistory({
  runnable: chain,
  getMessageHistory,
  inputMessagesKey: 'input',
  historyMessagesKey: 'chat_history',
});

// Limpeza automática de sessões inativas (30 minutos)
setInterval(() => {
  const now = new Date();
  for (const [sessionId, session] of sessions.entries()) {
    const inactiveTime = (now - session.lastActivity) / 1000 / 60; // em minutos
    if (inactiveTime > 30) {
      sessions.delete(sessionId);
      console.log(`Sessão ${sessionId} removida por inatividade`);
    }
  }
}, 5 * 60 * 1000); // Verificar a cada 5 minutos

// ROTAS DA API

// Criar nova sessão
app.post('/api/session/new', (req, res) => {
  const sessionId = uuidv4();
  sessions.set(sessionId, {
    messageHistory: new ChatMessageHistory(),
    systemPrompt: 'Você é um assistente útil e prestativo.',
    temperature: 0.5,
    createdAt: new Date(),
    lastActivity: new Date(),
  });
  
  console.log(`Nova sessão criada: ${sessionId}`);
  
  res.json({
    sessionId,
    message: 'Sessão criada com sucesso',
  });
});

// Enviar mensagem
app.post('/api/chat', async (req, res) => {
  try {
    const { sessionId, message, temperature, systemPrompt } = req.body;

    // Validações
    if (systemPrompt !== undefined && typeof systemPrompt !== 'string') {
      return res.status(400).json({ error: 'systemPrompt deve ser uma string' });
    }

    if (temperature !== undefined && (typeof temperature !== 'number' || temperature < 0 || temperature > 2)) {
      return res.status(400).json({ error: 'temperature deve ser um número entre 0 e 2' });
    }

    if (!sessionId) {
      return res.status(400).json({ error: 'sessionId é obrigatório' });
    }

    if (!message || message.trim() === '') {
      return res.status(400).json({ error: 'message não pode estar vazio' });
    }

    // Verificar se a sessão existe
    if (!sessions.has(sessionId)) {
      return res.status(404).json({ error: 'Sessão não encontrada ou expirada' });
    }

    const session = sessions.get(sessionId);

    // Atualizar system prompt se fornecido
    if (systemPrompt !== undefined) {
      session.systemPrompt = systemPrompt.trim() || 'Você é um assistente útil e prestativo.';
    }

    // Atualizar temperature se fornecida
    if (temperature !== undefined) {
      session.temperature = temperature;
    }

    console.log(`Mensagem recebida da sessão ${sessionId}: ${message}`);
    console.log(`Temperature: ${session.temperature}, System Prompt: ${session.systemPrompt.substring(0, 50)}...`);

    // Criar modelo específico para esta sessão se temperature foi alterada
    let currentChainWithHistory = chainWithHistory;

    if (session.temperature !== 0.5) {
      const customModel = new ChatOpenAI({
        modelName: 'gpt-4.1-nano',
        temperature: session.temperature,
        openAIApiKey: process.env.OPENAI_API_KEY,
      });

      const customChain = prompt.pipe(customModel);
      
      currentChainWithHistory = new RunnableWithMessageHistory({
        runnable: customChain,
        getMessageHistory,
        inputMessagesKey: 'input',
        historyMessagesKey: 'chat_history'
      });
    }

    // Processar mensagem com LangChain
    const response = await currentChainWithHistory.invoke(
      { 
        input: message,
        system_prompt: session.systemPrompt
      },
      { configurable: { sessionId } }
    );

    res.json({
      sessionId,
      response: response.content,
      timestamp: new Date(),
    });

  } catch (error) {
    console.error('Erro ao processar mensagem:', error);
    res.status(500).json({ 
      error: 'Erro ao processar mensagem',
      details: error.message 
    });
  }
});

// Obter histórico da sessão
app.get('/api/session/:sessionId/history', async (req, res) => {
  try {
    const { sessionId } = req.params;

    if (!sessions.has(sessionId)) {
      return res.status(404).json({ error: 'Sessão não encontrada' });
    }

    const session = sessions.get(sessionId);
    const messages = await session.messageHistory.getMessages();

    res.json({
      sessionId,
      messageCount: messages.length,
      messages: messages.map(msg => ({
        type: msg._getType(),
        content: msg.content,
      })),
      systemPrompt: session.systemPrompt,
      temperature: session.temperature,
      createdAt: session.createdAt,
      lastActivity: session.lastActivity,
    });

  } catch (error) {
    console.error('Erro ao obter histórico:', error);
    res.status(500).json({ error: 'Erro ao obter histórico' });
  }
});

// Limpar histórico de uma sessão
app.delete('/api/session/:sessionId/clear', async (req, res) => {
  try {
    const { sessionId } = req.params;

    if (!sessions.has(sessionId)) {
      return res.status(404).json({ error: 'Sessão não encontrada' });
    }

    const session = sessions.get(sessionId);
    await session.messageHistory.clear();

    res.json({
      message: 'Histórico limpo com sucesso',
      sessionId,
    });

  } catch (error) {
    console.error('Erro ao limpar histórico:', error);
    res.status(500).json({ error: 'Erro ao limpar histórico' });
  }
});

// Atualizar configurações da sessão
app.put('/api/session/:sessionId/config', (req, res) => {
  try {
    const { sessionId } = req.params;
    const { systemPrompt, temperature } = req.body;

    if (!sessions.has(sessionId)) {
      return res.status(404).json({ error: 'Sessão não encontrada' });
    }

    const session = sessions.get(sessionId);

    if (systemPrompt !== undefined) {
      if (typeof systemPrompt !== 'string') {
        return res.status(400).json({ error: 'systemPrompt deve ser uma string' });
      }
      session.systemPrompt = systemPrompt;
    }

    if (temperature !== undefined) {
      if (typeof temperature !== 'number' || temperature < 0 || temperature > 2) {
        return res.status(400).json({ error: 'temperature deve ser um número entre 0 e 2' });
      }
      session.temperature = temperature;
    }

    res.json({
      message: 'Configurações atualizadas com sucesso',
      sessionId,
      systemPrompt: session.systemPrompt,
      temperature: session.temperature,
    });

  } catch (error) {
    console.error('Erro ao atualizar configurações:', error);
    res.status(500).json({ error: 'Erro ao atualizar configurações' });
  }
});

// Deletar sessão
app.delete('/api/session/:sessionId', (req, res) => {
  const { sessionId } = req.params;

  if (!sessions.has(sessionId)) {
    return res.status(404).json({ error: 'Sessão não encontrada' });
  }

  sessions.delete(sessionId);
  console.log(`Sessão deletada: ${sessionId}`);

  res.json({
    message: 'Sessão deletada com sucesso',
    sessionId,
  });
});

// Status do servidor
app.get('/api/status', (req, res) => {
  res.json({
    status: 'online',
    activeSessions: sessions.size,
    uptime: process.uptime(),
  });
});

// Iniciar servidor
app.listen(PORT, '0.0.0.0', () => {
  const ifaces = require('os').networkInterfaces();
  let localIP = 'localhost';
  
  // Encontrar IP local
  Object.keys(ifaces).forEach(ifname => {
    ifaces[ifname].forEach(iface => {
      if (iface.family === 'IPv4' && !iface.internal) {
        localIP = iface.address;
      }
    });
  });
  
  console.log(`\n🤖 Servidor Chatbot iniciado na porta ${PORT}`);
  console.log(`📍 Acesso local: http://localhost:${PORT}`);
  console.log(`🌐 Acesso rede: http://${localIP}:${PORT}`);
  console.log(`📊 Status: http://localhost:${PORT}/api/status`);
  console.log(`📘 Swagger UI: http://localhost:${PORT}/api-docs`);
});

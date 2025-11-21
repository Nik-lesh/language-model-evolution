import React, { useState, useRef, useEffect } from "react";
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  Container,
  CircularProgress,
  Slider,
  Chip,
  Divider,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import DeleteIcon from "@mui/icons-material/Delete";
import axios from "axios";

const API_URL = "http://localhost:8000";

function ChatInterface() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [maxLength, setMaxLength] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setLoading(true);

    try {
      // Add user message
      const userMessage = {
        role: "user",
        content: prompt,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Call API
      const response = await axios.post(`${API_URL}/generate`, {
        prompt: prompt,
        max_length: maxLength,
        temperature: temperature,
      });

      // Add AI response
      const aiMessage = {
        role: "assistant",
        content: response.data.generated_text,
        timestamp: new Date(),
        metadata: response.data.model_info,
      };
      setMessages((prev) => [...prev, aiMessage]);

      // Clear input
      setPrompt("");
    } catch (error) {
      console.error("Error:", error);

      const errorMessage = {
        role: "error",
        content: `Error: ${
          error.response?.data?.detail ||
          error.message ||
          "Failed to generate text"
        }`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const examplePrompts = [
    "How do I start investing with $1000?",
    "What is compound interest?",
    "Should I invest in index funds?",
    "Explain the stock market",
  ];

  const handleExampleClick = (example) => {
    setPrompt(example);
  };

  return (
    <Container
      maxWidth="lg"
      sx={{ py: 4, height: "100vh", display: "flex", flexDirection: "column" }}
    >
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography
          variant="h3"
          component="h1"
          gutterBottom
          align="center"
          sx={{
            fontWeight: 700,
            background: "linear-gradient(45deg, grey, grey 50%, black 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          🤖 Financial Advisor AI
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary">
          Powered by Transformer trained on 1GB financial corpus (168M words)
        </Typography>
      </Box>

      <Box sx={{ display: "flex", gap: 2, flex: 1, overflow: "hidden" }}>
        {/* Left Panel - Settings */}
        <Paper
          elevation={3}
          sx={{ p: 3, width: 280, display: "flex", flexDirection: "column" }}
        >
          <Typography variant="h6" gutterBottom>
            ⚙️ Settings
          </Typography>

          <Divider sx={{ my: 2 }} />

          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" gutterBottom fontWeight={600}>
              Length: {maxLength} words
            </Typography>
            <Slider
              value={maxLength}
              onChange={(e, val) => setMaxLength(val)}
              min={20}
              max={200}
              step={10}
              valueLabelDisplay="auto"
              size="small"
            />
          </Box>

          <Box sx={{ mb: 3 }}>
            <Typography variant="body2" gutterBottom fontWeight={600}>
              Creativity: {temperature}
            </Typography>
            <Slider
              value={temperature}
              onChange={(e, val) => setTemperature(val)}
              min={0.3}
              max={1.5}
              step={0.1}
              valueLabelDisplay="auto"
              size="small"
            />
            <Typography variant="caption" color="text.secondary">
              Higher = more creative, Lower = more conservative
            </Typography>
          </Box>

          <Divider sx={{ my: 2 }} />

          <Typography variant="body2" gutterBottom fontWeight={600}>
            💡 Try these prompts:
          </Typography>
          {examplePrompts.map((example, idx) => (
            <Chip
              key={idx}
              label={example}
              onClick={() => handleExampleClick(example)}
              sx={{
                mb: 1,
                cursor: "pointer",
                width: "100%",
                justifyContent: "flex-start",
              }}
              size="small"
            />
          ))}

          <Box sx={{ mt: "auto", pt: 2 }}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<DeleteIcon />}
              onClick={clearChat}
              disabled={messages.length === 0}
            >
              Clear Chat
            </Button>
          </Box>

          <Typography variant="caption" sx={{ mt: 2 }} color="text.secondary">
            🏆 Best Model: Char-level LSTM (1.47 loss)
            <br />
            📊 Current: Word Transformer (4.01 loss)
          </Typography>
        </Paper>

        {/* Right Panel - Chat */}
        <Box
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
          }}
        >
          {/* Messages */}
          <Paper
            elevation={3}
            sx={{
              flex: 1,
              p: 3,
              mb: 2,
              overflow: "auto",
              bgcolor: "#fafafa",
            }}
          >
            {messages.length === 0 ? (
              <Box sx={{ textAlign: "center", py: 12 }}>
                <Typography variant="h5" gutterBottom color="text.secondary">
                  👋 Welcome to Financial Advisor AI
                </Typography>
                <Typography
                  variant="body1"
                  color="text.secondary"
                  sx={{ mb: 3 }}
                >
                  Ask me anything about investing, finance, or the stock market!
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Built with custom-trained language models:
                  <br />
                  RNN → LSTM → Transformer evolution
                </Typography>
              </Box>
            ) : (
              <>
                {messages.map((msg, idx) => (
                  <Box
                    key={idx}
                    sx={{
                      mb: 2,
                      display: "flex",
                      justifyContent:
                        msg.role === "user" ? "flex-end" : "flex-start",
                    }}
                  >
                    <Paper
                      elevation={msg.role === "error" ? 4 : 2}
                      sx={{
                        p: 2,
                        maxWidth: "75%",
                        bgcolor:
                          msg.role === "user"
                            ? "primary.main"
                            : msg.role === "error"
                            ? "error.main"
                            : "white",
                        color:
                          msg.role === "user" || msg.role === "error"
                            ? "white"
                            : "black",
                        borderRadius: 2,
                      }}
                    >
                      <Typography
                        variant="caption"
                        display="block"
                        sx={{ mb: 0.5, opacity: 0.8, fontWeight: 600 }}
                      >
                        {msg.role === "user"
                          ? "👤 You"
                          : msg.role === "error"
                          ? "⚠️ Error"
                          : "🤖 AI"}
                      </Typography>
                      <Typography
                        variant="body1"
                        sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}
                      >
                        {msg.content}
                      </Typography>
                      {msg.metadata && (
                        <Typography
                          variant="caption"
                          display="block"
                          sx={{ mt: 1, opacity: 0.7 }}
                        >
                          {msg.metadata.val_loss} loss •{" "}
                          {msg.metadata.parameters}
                        </Typography>
                      )}
                    </Paper>
                  </Box>
                ))}
                {loading && (
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 2,
                      ml: 2,
                    }}
                  >
                    <CircularProgress size={20} />
                    <Typography variant="body2" color="text.secondary">
                      AI is thinking...
                    </Typography>
                  </Box>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </Paper>

          {/* Input */}
          <Paper elevation={3} sx={{ p: 2 }}>
            <Box sx={{ display: "flex", gap: 1.5 }}>
              <TextField
                fullWidth
                multiline
                maxRows={4}
                placeholder="Ask about investing, stocks, personal finance..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading}
                variant="outlined"
                size="medium"
              />
              <Button
                variant="contained"
                endIcon={<SendIcon />}
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                sx={{ minWidth: 120, height: 56 }}
                size="large"
              >
                Send
              </Button>
            </Box>
          </Paper>
        </Box>
      </Box>
    </Container>
  );
}

export default ChatInterface;

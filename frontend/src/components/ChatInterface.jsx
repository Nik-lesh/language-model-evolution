import React, { useState, useRef, useEffect } from "react";
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
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
      const userMessage = {
        role: "user",
        content: prompt,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);

      const response = await axios.post(`${API_URL}/generate`, {
        prompt: prompt,
        max_length: maxLength,
        temperature: temperature,
      });

      const aiMessage = {
        role: "assistant",
        content: response.data.generated_text,
        timestamp: new Date(),
        metadata: response.data.model_info,
      };
      setMessages((prev) => [...prev, aiMessage]);

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
    <Box
      sx={{
        width: "100vw",
        height: "100vh",
        bgcolor: "#000000",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        p: 3,
      }}
    >
      {/* Header */}
      <Box sx={{ mb: 2, flexShrink: 0 }}>
        <Typography
          variant="h3"
          component="h1"
          gutterBottom
          align="center"
          sx={{
            fontWeight: 700,
            background:
              "linear-gradient(45deg, #9e9e9e, #bdbdbd 50%, #e0e0e0 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            mb: 1,
          }}
        >
          Financial Advisor AI
        </Typography>
        <Typography
          variant="subtitle1"
          align="center"
          sx={{ color: "#b0b0b0" }}
        >
          Powered by Transformer trained on 1GB financial corpus (168M words)
        </Typography>
      </Box>

      <Box sx={{ display: "flex", gap: 2, flex: 1, minHeight: 0 }}>
        {/* Left Panel - Settings */}
        <Paper
          elevation={3}
          sx={{
            p: 3,
            width: 300,
            display: "flex",
            flexDirection: "column",
            bgcolor: "#1a1a1a",
            color: "#e0e0e0",
            flexShrink: 0,
          }}
        >
          <Typography variant="h6" gutterBottom sx={{ color: "#e0e0e0" }}>
            ⚙️ Settings
          </Typography>

          <Divider sx={{ my: 2, bgcolor: "#404040" }} />

          <Box sx={{ mb: 3 }}>
            <Typography
              variant="body2"
              gutterBottom
              fontWeight={600}
              sx={{ color: "#e0e0e0" }}
            >
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
              sx={{
                color: "#9e9e9e",
                "& .MuiSlider-thumb": {
                  bgcolor: "#bdbdbd",
                },
              }}
            />
          </Box>

          <Box sx={{ mb: 3 }}>
            <Typography
              variant="body2"
              gutterBottom
              fontWeight={600}
              sx={{ color: "#e0e0e0" }}
            >
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
              sx={{
                color: "#9e9e9e",
                "& .MuiSlider-thumb": {
                  bgcolor: "#bdbdbd",
                },
              }}
            />
            <Typography variant="caption" sx={{ color: "#909090" }}>
              Higher = more creative, Lower = more conservative
            </Typography>
          </Box>

          <Divider sx={{ my: 2, bgcolor: "#404040" }} />

          <Typography
            variant="body2"
            gutterBottom
            fontWeight={600}
            sx={{ color: "#e0e0e0" }}
          >
            💡 Try these prompts:
          </Typography>
          <Box sx={{ flex: 1, overflow: "auto", minHeight: 0 }}>
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
                  bgcolor: "#2a2a2a",
                  color: "#d0d0d0",
                  "&:hover": {
                    bgcolor: "#353535",
                  },
                }}
                size="small"
              />
            ))}
          </Box>

          <Box sx={{ pt: 2, flexShrink: 0 }}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<DeleteIcon />}
              onClick={clearChat}
              disabled={messages.length === 0}
              sx={{
                borderColor: "#505050",
                color: "#d0d0d0",
                "&:hover": {
                  borderColor: "#707070",
                  bgcolor: "#2a2a2a",
                },
                "&.Mui-disabled": {
                  borderColor: "#303030",
                  color: "#606060",
                },
              }}
            >
              Clear Chat
            </Button>
          </Box>

          <Typography
            variant="caption"
            sx={{ mt: 2, color: "#909090", flexShrink: 0 }}
          >
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
            minHeight: 0,
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
              bgcolor: "#0a0a0a",
              minHeight: 0,
            }}
          >
            {messages.length === 0 ? (
              <Box
                sx={{
                  textAlign: "center",
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                }}
              >
                <Typography variant="h5" gutterBottom sx={{ color: "#c0c0c0" }}>
                  👋 Welcome to Financial Advisor AI
                </Typography>
                <Typography variant="body1" sx={{ mb: 3, color: "#a0a0a0" }}>
                  Ask me anything about investing, finance, or the stock market!
                </Typography>
                <Typography variant="body2" sx={{ color: "#909090" }}>
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
                            ? "#2a2a2a"
                            : msg.role === "error"
                            ? "#4a1a1a"
                            : "#1a1a1a",
                        color: "#e0e0e0",
                        borderRadius: 2,
                      }}
                    >
                      <Typography
                        variant="caption"
                        display="block"
                        sx={{
                          mb: 0.5,
                          opacity: 0.8,
                          fontWeight: 600,
                          color: "#c0c0c0",
                        }}
                      >
                        {msg.role === "user"
                          ? "👤 You"
                          : msg.role === "error"
                          ? "⚠️ Error"
                          : "🤖 AI"}
                      </Typography>
                      <Typography
                        variant="body1"
                        sx={{
                          whiteSpace: "pre-wrap",
                          lineHeight: 1.6,
                          color: "#e0e0e0",
                        }}
                      >
                        {msg.content}
                      </Typography>
                      {msg.metadata && (
                        <Typography
                          variant="caption"
                          display="block"
                          sx={{ mt: 1, opacity: 0.7, color: "#a0a0a0" }}
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
                    <CircularProgress size={20} sx={{ color: "#9e9e9e" }} />
                    <Typography variant="body2" sx={{ color: "#a0a0a0" }}>
                      AI is thinking...
                    </Typography>
                  </Box>
                )}
                <div ref={messagesEndRef} />
              </>
            )}
          </Paper>

          {/* Input */}
          <Paper elevation={3} sx={{ p: 2, bgcolor: "#1a1a1a", flexShrink: 0 }}>
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
                sx={{
                  "& .MuiOutlinedInput-root": {
                    color: "#e0e0e0",
                    "& fieldset": {
                      borderColor: "#404040",
                    },
                    "&:hover fieldset": {
                      borderColor: "#606060",
                    },
                    "&.Mui-focused fieldset": {
                      borderColor: "#808080",
                    },
                  },
                  "& .MuiInputBase-input::placeholder": {
                    color: "#808080",
                    opacity: 1,
                  },
                }}
              />
              <Button
                variant="contained"
                endIcon={<SendIcon />}
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                sx={{
                  minWidth: 120,
                  height: 56,
                  bgcolor: "#2a2a2a",
                  color: "#e0e0e0",
                  "&:hover": {
                    bgcolor: "#353535",
                  },
                  "&.Mui-disabled": {
                    bgcolor: "#1a1a1a",
                    color: "#606060",
                  },
                }}
                size="large"
              >
                Send
              </Button>
            </Box>
          </Paper>
        </Box>
      </Box>
    </Box>
  );
}

export default ChatInterface;

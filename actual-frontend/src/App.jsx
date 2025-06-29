import { useState, useRef, useEffect, useCallback } from "react";
import { BounceLoader } from "react-spinners";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState("dark");
  const [conversationHistory, setConversationHistory] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const abortControllerRef = useRef(null);
  
  const synth = typeof window !== 'undefined' ? window.speechSynthesis : null;

  // Generate conversation ID
  const generateConversationId = () => {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  };

  // Save conversation to history
  const saveConversation = useCallback((messages, conversationId) => {
    if (messages.length === 0) return;
    
    const conversation = {
      id: conversationId,
      title: messages[0]?.content?.substring(0, 50) + (messages[0]?.content?.length > 50 ? "..." : ""),
      messages: messages,
      timestamp: new Date().toISOString(),
      messageCount: messages.length
    };

    setConversationHistory(prev => {
      const filtered = prev.filter(conv => conv.id !== conversationId);
      const updated = [conversation, ...filtered].slice(0, 5); // Keep only 5 conversations
      return updated;
    });
  }, []);

  // Load conversation from history
  const loadConversation = useCallback((conversationId) => {
    const conversation = conversationHistory.find(conv => conv.id === conversationId);
    if (conversation) {
      setMessages(conversation.messages);
      setCurrentConversationId(conversationId);
    }
  }, [conversationHistory]);

  // Start new conversation
  const startNewConversation = useCallback(() => {
    if (messages.length > 0) {
      saveConversation(messages, currentConversationId);
    }
    setMessages([]);
    setCurrentConversationId(generateConversationId());
  }, [messages, currentConversationId, saveConversation]);

  // Initialize first conversation
  useEffect(() => {
    if (!currentConversationId) {
      setCurrentConversationId(generateConversationId());
    }
  }, [currentConversationId]);

  const speakText = useCallback((text) => {
    if (!synth) return;
    synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    const voices = synth.getVoices();
    const preferredVoice = voices.find(voice => 
      voice.lang.includes('en') && voice.name.includes('Female')
    );
    
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    
    synth.speak(utterance);
  }, [synth]);

  useEffect(() => {
    if (synth) {
      synth.onvoiceschanged = () => {
        synth.getVoices();
      };
    }
    
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [synth]);

  const sendMessage = useCallback(async () => {
    if (!input.trim()) return;
    setError(null);

    const userMessage = { role: "user", content: input, isComplete: true };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setIsLoading(true);

    // Add placeholder for assistant response
    const assistantMessage = { role: "assistant", content: "", isComplete: false };
    setMessages(prev => [...prev, assistantMessage]);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: input, 
          history: updatedMessages.map(msg => ({ 
            role: msg.role, 
            content: msg.content 
          })),
          model: "mistral"
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let streamedContent = "";
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              if (data.content) {
                streamedContent += data.content;
                console.log('Streamed so far:', streamedContent); // DEBUG
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = {
                    ...newMessages[newMessages.length - 1],
                    content: streamedContent,
                    isComplete: false
                  };
                  return newMessages;
                });
              }
            } catch (error) {
              console.warn('Error parsing stream chunk', error);
            }
          }
        }
      }
      
      // Mark message as complete
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          content: streamedContent || "I'm here for you. Your feelings matter, and you're not alone.",
          isComplete: true
        };
        return newMessages;
      });
      
      // Also log the final content for debugging
      console.log('Final streamed content:', streamedContent);
      
      // Auto-speak if enabled
      if (autoSpeak && streamedContent) {
        speakText(streamedContent);
      }
      
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error("Error:", error);
        setError("Sorry, I'm having trouble connecting right now. Please try again.");
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = { 
            role: "assistant", 
            content: "I'm sorry, I'm having trouble responding right now. Please try again in a moment.",
            isComplete: true 
          };
          return newMessages;
        });
      }
    } finally {
      setIsLoading(false);
    }
  }, [input, messages, autoSpeak, speakText]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const toggleTheme = () => {
    setTheme(prev => prev === "dark" ? "light" : "dark");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && input.trim()) {
        sendMessage();
      }
    }
  };

  const formatMessageContent = (content) => {
    if (!content) return "";
    return content.toString().replace(/\n/g, '<br />');
  };

  const stopSpeaking = () => {
    if (synth) {
      synth.cancel();
      setIsSpeaking(false);
    }
  };

  const toggleAutoSpeak = () => {
    setAutoSpeak(prev => !prev);
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now - date) / (1000 * 60 * 60);
    
    if (diffInHours < 1) {
      return "Just now";
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className={`flex h-screen transition-colors duration-300 ${
      theme === "dark" ? "bg-gray-900 text-white" : "bg-gray-50 text-gray-800"
    }`}>
      {/* Sidebar */}
      <div className={`${
        sidebarOpen ? "w-80" : "w-0"
      } transition-all duration-300 overflow-hidden ${
        theme === "dark" ? "bg-gray-800 border-r border-gray-700" : "bg-white border-r border-gray-200"
      }`}>
        <div className="p-4">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold">Conversations</h2>
            <button
              onClick={startNewConversation}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition ${
                theme === "dark" 
                  ? "bg-blue-600 hover:bg-blue-700 text-white" 
                  : "bg-blue-500 hover:bg-blue-600 text-white"
              }`}
            >
              New Chat
            </button>
          </div>
          
          <div className="space-y-2">
            {conversationHistory.map((conversation) => (
              <div
                key={conversation.id}
                onClick={() => loadConversation(conversation.id)}
                className={`p-3 rounded-lg cursor-pointer transition ${
                  currentConversationId === conversation.id
                    ? theme === "dark" 
                      ? "bg-blue-600 text-white" 
                      : "bg-blue-500 text-white"
                    : theme === "dark"
                      ? "bg-gray-700 hover:bg-gray-600"
                      : "bg-gray-100 hover:bg-gray-200"
                }`}
              >
                <div className="font-medium text-sm mb-1 truncate">
                  {conversation.title}
                </div>
                <div className="flex justify-between items-center text-xs opacity-75">
                  <span>{conversation.messageCount} messages</span>
                  <span>{formatTimestamp(conversation.timestamp)}</span>
                </div>
              </div>
            ))}
            
            {conversationHistory.length === 0 && (
              <div className={`text-center py-8 text-sm ${
                theme === "dark" ? "text-gray-400" : "text-gray-500"
              }`}>
                No previous conversations
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className={`flex items-center justify-between p-4 border-b ${
          theme === "dark" ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
        }`}>
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`p-2 rounded-lg ${
                theme === "dark" ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-100 hover:bg-gray-200"
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-xl font-bold">AI Therapy Assistant</h1>
          </div>
          
          <div className="flex items-center space-x-2">
            <button 
              onClick={toggleAutoSpeak}
              className={`p-2 rounded-lg transition ${
                autoSpeak 
                  ? theme === "dark" ? "bg-green-600" : "bg-green-500" 
                  : theme === "dark" ? "bg-gray-700" : "bg-gray-200"
              }`}
              title={autoSpeak ? "Auto-speak enabled" : "Auto-speak disabled"}
            >
              {autoSpeak ? "🔊" : "🔈"}
            </button>
            <button 
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition ${
                theme === "dark" ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-100 hover:bg-gray-200"
              }`}
            >
              {theme === "dark" ? "🌙" : "☀️"}
            </button>
          </div>
        </div>
        
        {/* Chat Messages */}
        <div 
          ref={chatContainerRef} 
          className={`flex-1 overflow-y-auto p-4 space-y-4 ${
            theme === "dark" ? "bg-gray-900" : "bg-gray-50"
          }`}
        >
          {messages.length === 0 && (
            <div className={`text-center py-12 ${
              theme === "dark" ? "text-gray-400" : "text-gray-500"
            }`}>
              <div className="text-6xl mb-4">🧠</div>
              <h3 className="text-lg font-medium mb-2">Welcome to AI Therapy</h3>
              <p className="text-sm">Start a conversation to get mental health support</p>
            </div>
          )}
          
          {messages.map((msg, index) => (
            <div 
              key={index} 
              className={`flex items-start space-x-3 ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.role === "assistant" && (
                <div className="flex-shrink-0">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    theme === "dark" ? "bg-green-500" : "bg-green-400"
                  }`}>
                    <span className="text-white font-bold text-sm">AI</span>
                  </div>
                </div>
              )}
              
              <div
                className={`p-4 rounded-2xl max-w-[70%] shadow-sm ${
                  msg.role === "user"
                    ? theme === "dark" 
                      ? "bg-blue-600 text-white" 
                      : "bg-blue-500 text-white"
                    : theme === "dark"
                      ? "bg-gray-700 text-white"
                      : "bg-white text-gray-800 border border-gray-200"
                }`}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                  {msg.content && msg.content.trim()
                    ? formatMessageContent(msg.content.toString())
                    : "I'm here for you. Your feelings matter, and you're not alone."}
                </div>
                
                {msg.role === "assistant" && msg.isComplete && (
                  <div className="mt-3 flex justify-end">
                    {isSpeaking && msg === messages[messages.length - 1] ? (
                      <button
                        onClick={stopSpeaking}
                        className={`text-xs px-3 py-1 rounded-full transition ${
                          theme === "dark" ? "bg-red-600 hover:bg-red-700" : "bg-red-500 hover:bg-red-600"
                        } text-white`}
                      >
                        Stop
                      </button>
                    ) : (
                      <button
                        onClick={() => speakText(msg.content)}
                        className={`text-xs px-3 py-1 rounded-full transition ${
                          theme === "dark" ? "bg-gray-600 hover:bg-gray-500" : "bg-gray-200 hover:bg-gray-300"
                        }`}
                      >
                        🔊
                      </button>
                    )}
                  </div>
                )}
              </div>
              
              {msg.role === "user" && (
                <div className="flex-shrink-0">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    theme === "dark" ? "bg-blue-500" : "bg-blue-400"
                  }`}>
                    <span className="text-white font-bold text-sm">You</span>
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-center items-center p-4">
              <BounceLoader color={theme === "dark" ? "#4ade80" : "#22c55e"} size={30} />
            </div>
          )}
          
          {error && (
            <div className="bg-red-500 text-white p-3 rounded-lg text-center text-sm">
              {error}
            </div>
          )}
        </div>
        
        {/* Input Area */}
        <div className={`p-4 border-t ${
          theme === "dark" ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
        }`}>
          <div className="flex space-x-3">
            <textarea
              ref={inputRef}
              rows="2"
              className={`flex-1 p-3 rounded-xl resize-none transition ${
                theme === "dark" 
                  ? "text-white bg-gray-700 border border-gray-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500" 
                  : "text-gray-800 bg-gray-50 border border-gray-300 focus:border-blue-400 focus:ring-2 focus:ring-blue-400"
              }`}
              placeholder="Share what's on your mind... (Shift+Enter for new line)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <button
              className={`px-6 py-3 rounded-xl font-medium transition ${
                isLoading
                  ? theme === "dark" ? "bg-gray-600" : "bg-gray-300"
                  : theme === "dark" 
                    ? "bg-blue-600 hover:bg-blue-700" 
                    : "bg-blue-500 hover:bg-blue-600"
              } text-white cursor-${isLoading ? "not-allowed" : "pointer"}`}
              onClick={sendMessage}
              disabled={isLoading || !input.trim()}
            >
              {isLoading ? "Sending..." : "Send"}
            </button>
          </div>
          
          <div className="mt-3 flex items-center justify-between">
            <div className="text-xs text-gray-500">
              Crisis? Call 988
            </div>
            <div className="text-xs text-gray-500">
              Powered by AI Therapy
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
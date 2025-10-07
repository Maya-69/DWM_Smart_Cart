import React, { useState, useEffect } from 'react';
import { ShoppingCart, TrendingUp, X, Plus, Minus, Brain, Filter, ChevronDown } from 'lucide-react';

const API_URL = 'http://localhost:5000/api';

const App = () => {
  const [products, setProducts] = useState([]);
  const [cart, setCart] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [aiInsights, setAiInsights] = useState(null);
  const [showGraph, setShowGraph] = useState(false);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [sortBy, setSortBy] = useState('popular');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchProducts();
    fetchCategories();
  }, []);

  useEffect(() => {
    if (cart.length > 0) {
      fetchRecommendations();
    } else {
      setRecommendations([]);
      setAiInsights(null);
    }
  }, [cart]);

  const fetchProducts = async () => {
    try {
      const response = await fetch(`${API_URL}/products`);
      const data = await response.json();
      setProducts(data);
    } catch (error) {
      console.error('Error fetching products:', error);
    }
  };

  const fetchCategories = async () => {
    try {
      const response = await fetch(`${API_URL}/categories`);
      const data = await response.json();
      setCategories(data);
    } catch (error) {
      console.error('Error fetching categories:', error);
    }
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    try {
      const cartIds = cart.map(item => item.id);
      const response = await fetch(`${API_URL}/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cart_items: cartIds })
      });
      const data = await response.json();
      setRecommendations(data.recommendations || []);
      setAiInsights(data.ai_insights || null);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const addToCart = (product) => {
    const existing = cart.find(item => item.id === product.id);
    if (existing) {
      setCart(cart.map(item => 
        item.id === product.id ? { ...item, quantity: item.quantity + 1 } : item
      ));
    } else {
      setCart([...cart, { ...product, quantity: 1 }]);
    }
  };

  const removeFromCart = (productId) => {
    const existing = cart.find(item => item.id === productId);
    if (existing.quantity === 1) {
      setCart(cart.filter(item => item.id !== productId));
    } else {
      setCart(cart.map(item => 
        item.id === productId ? { ...item, quantity: item.quantity - 1 } : item
      ));
    }
  };

  const deleteFromCart = (productId) => {
    setCart(cart.filter(item => item.id !== productId));
  };

  const getCartTotal = () => {
    return cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  };

  const formatNumber = (num) => {
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num;
  };

  const getFilteredProducts = () => {
    let filtered = selectedCategory === 'all' 
      ? products 
      : products.filter(p => p.category === selectedCategory);
    
    if (sortBy === 'popular') {
      filtered = [...filtered].sort((a, b) => b.total_purchases - a.total_purchases);
    } else if (sortBy === 'price_low') {
      filtered = [...filtered].sort((a, b) => a.price - b.price);
    } else if (sortBy === 'price_high') {
      filtered = [...filtered].sort((a, b) => b.price - a.price);
    }
    
    return filtered;
  };

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', padding: '24px' }}>
      <style>{`
        @keyframes slideDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.8; } }
        @keyframes shimmer { 0% { background-position: -1000px 0; } 100% { background-position: 1000px 0; } }
        .card-hover { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
        .card-hover:hover { transform: translateY(-4px); box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3); }
        .btn-hover { transition: all 0.2s ease; }
        .btn-hover:hover { transform: scale(1.02); filter: brightness(1.1); }
        .btn-hover:active { transform: scale(0.98); }
        .gradient-text { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .glass { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .loading { background: linear-gradient(90deg, #1e293b 0%, #334155 50%, #1e293b 100%); background-size: 1000px 100%; animation: shimmer 2s infinite; }
      `}</style>

      <div style={{ maxWidth: '1600px', margin: '0 auto' }}>
        {/* Header */}
        <div className="glass" style={{ 
          borderRadius: '20px', 
          padding: '28px 40px', 
          marginBottom: '24px', 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          animation: 'slideDown 0.6s ease-out',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
        }}>
          <div>
            <h1 className="gradient-text" style={{ margin: 0, fontSize: '36px', fontWeight: '900', letterSpacing: '-1px' }}>
              SmartCart AI
            </h1>
            <p style={{ margin: '8px 0 0 0', color: '#94a3b8', fontSize: '15px', fontWeight: '500' }}>
              Advanced ML-powered shopping intelligence
            </p>
          </div>
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <button
              onClick={() => setShowGraph(!showGraph)}
              className="btn-hover"
              disabled={cart.length === 0}
              style={{ 
                background: cart.length > 0 ? 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' : 'rgba(255,255,255,0.1)', 
                border: 'none', 
                padding: '14px 28px', 
                borderRadius: '14px', 
                cursor: cart.length > 0 ? 'pointer' : 'not-allowed', 
                display: 'flex', 
                alignItems: 'center', 
                gap: '10px', 
                fontWeight: '700', 
                color: 'white', 
                fontSize: '15px',
                opacity: cart.length > 0 ? 1 : 0.4
              }}
            >
              <Brain size={20} />
              AI Insights
            </button>
            <div style={{ 
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 
              color: 'white', 
              padding: '16px 32px', 
              borderRadius: '16px', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '14px', 
              fontWeight: '700',
              boxShadow: '0 8px 24px rgba(99, 102, 241, 0.4)'
            }}>
              <ShoppingCart size={24} />
              <span style={{ fontSize: '16px' }}>{cart.length} items</span>
              <div style={{ width: '2px', height: '24px', background: 'rgba(255,255,255,0.3)' }} />
              <span style={{ fontSize: '20px', fontWeight: '900' }}>₹{getCartTotal()}</span>
            </div>
          </div>
        </div>

        {/* AI Insights Graph */}
        {showGraph && aiInsights && (
          <div className="glass" style={{ 
            borderRadius: '20px', 
            padding: '32px', 
            marginBottom: '24px',
            animation: 'slideUp 0.5s ease-out',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
              <Brain size={28} style={{ color: '#6366f1' }} />
              <h3 style={{ margin: 0, fontSize: '24px', fontWeight: '900', color: 'white' }}>
                AI Intent Analysis
              </h3>
            </div>
            
            {/* Detected Intent */}
            <div style={{ 
              background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 
              borderRadius: '16px', 
              padding: '24px',
              marginBottom: '24px'
            }}>
              <div style={{ fontSize: '14px', color: 'rgba(255,255,255,0.8)', fontWeight: '600', marginBottom: '8px' }}>
                🎯 DETECTED INTENT
              </div>
              <div style={{ fontSize: '28px', fontWeight: '900', color: 'white', marginBottom: '12px' }}>
                {aiInsights.intent}
              </div>
              <div style={{ fontSize: '15px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.6' }}>
                {aiInsights.reasoning}
              </div>
              <div style={{ 
                display: 'flex', 
                gap: '12px', 
                marginTop: '16px',
                flexWrap: 'wrap'
              }}>
                <div style={{ 
                  background: 'rgba(255,255,255,0.2)', 
                  padding: '8px 16px', 
                  borderRadius: '10px',
                  fontSize: '14px',
                  fontWeight: '700',
                  color: 'white'
                }}>
                  Confidence: {aiInsights.confidence}%
                </div>
                {aiInsights.categories.map((cat, i) => (
                  <div key={i} style={{ 
                    background: 'rgba(255,255,255,0.15)', 
                    padding: '8px 16px', 
                    borderRadius: '10px',
                    fontSize: '13px',
                    fontWeight: '600',
                    color: 'white'
                  }}>
                    #{cat}
                  </div>
                ))}
              </div>
            </div>

            {/* Graph Visualization */}
            <div style={{ 
              background: 'rgba(15, 23, 42, 0.6)', 
              borderRadius: '16px', 
              padding: '32px',
              border: '1px solid rgba(99, 102, 241, 0.3)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-start', gap: '40px', flexWrap: 'wrap' }}>
                {/* Cart Items */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
                  <div style={{ 
                    fontSize: '12px', 
                    fontWeight: '800', 
                    color: '#6366f1', 
                    textTransform: 'uppercase',
                    letterSpacing: '1.5px'
                  }}>
                    YOUR CART
                  </div>
                  {cart.map((item, idx) => (
                    <div key={item.id} style={{ 
                      background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)', 
                      padding: '16px 24px', 
                      borderRadius: '14px',
                      border: '2px solid #6366f1',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      minWidth: '200px',
                      animation: `fadeIn 0.4s ease-out ${idx * 0.1}s forwards`,
                      opacity: 0
                    }}>
                      <span style={{ fontSize: '32px' }}>{item.img}</span>
                      <div>
                        <div style={{ fontSize: '15px', fontWeight: '700', color: 'white' }}>{item.name}</div>
                        <div style={{ fontSize: '13px', color: '#94a3b8', fontWeight: '600' }}>Qty: {item.quantity}</div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* AI Recommendations */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
                  <div style={{ 
                    fontSize: '12px', 
                    fontWeight: '800', 
                    color: '#8b5cf6', 
                    textTransform: 'uppercase',
                    letterSpacing: '1.5px'
                  }}>
                    AI SUGGESTS
                  </div>
                  {recommendations.slice(0, 4).map((rec, idx) => (
                    <div key={rec.id} style={{ position: 'relative' }}>
                      <div style={{
                        position: 'absolute',
                        left: '-50px',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        width: '40px',
                        height: '2px',
                        background: `linear-gradient(to right, #6366f1, #8b5cf6)`,
                        opacity: 0.6
                      }} />
                      <div style={{ 
                        background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)', 
                        padding: '16px 24px', 
                        borderRadius: '14px',
                        border: '2px solid #8b5cf6',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        minWidth: '220px',
                        animation: `fadeIn 0.4s ease-out ${0.4 + idx * 0.1}s forwards`,
                        opacity: 0
                      }}>
                        <span style={{ fontSize: '32px' }}>{rec.img}</span>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: '15px', fontWeight: '700', color: 'white', marginBottom: '4px' }}>
                            {rec.name}
                          </div>
                          <div style={{ fontSize: '12px', color: '#8b5cf6', fontWeight: '700' }}>
                            {rec.ai_match}% AI Match
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* AI Recommendations */}
        {recommendations.length > 0 && (
          <div style={{ 
            background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 
            borderRadius: '20px', 
            padding: '32px', 
            marginBottom: '24px',
            boxShadow: '0 20px 60px rgba(99, 102, 241, 0.4)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{ 
              position: 'absolute', 
              top: '-100px', 
              right: '-100px', 
              width: '300px', 
              height: '300px', 
              background: 'rgba(255,255,255,0.1)', 
              borderRadius: '50%',
              animation: 'pulse 4s ease-in-out infinite'
            }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '28px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Brain size={32} color="white" />
                  <div>
                    <h2 style={{ margin: 0, color: 'white', fontSize: '28px', fontWeight: '900', letterSpacing: '-0.5px' }}>
                      AI Recommendations
                    </h2>
                    <p style={{ margin: '4px 0 0 0', color: 'rgba(255,255,255,0.8)', fontSize: '14px' }}>
                      Based on {aiInsights?.intent || 'your selection'}
                    </p>
                  </div>
                </div>
                {loading && <div className="loading" style={{ width: '100px', height: '4px', borderRadius: '2px' }} />}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '20px' }}>
                {recommendations.map((rec, idx) => (
                  <div key={rec.id} style={{ 
                    background: 'rgba(255,255,255,0.95)', 
                    borderRadius: '18px', 
                    padding: '24px',
                    animation: `fadeIn 0.5s ease-out ${idx * 0.1}s forwards`,
                    opacity: 0
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                      <span style={{ fontSize: '64px' }}>{rec.img}</span>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', alignItems: 'flex-end' }}>
                        <div style={{ 
                          background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 
                          color: 'white', 
                          padding: '6px 14px', 
                          borderRadius: '10px', 
                          fontSize: '14px', 
                          fontWeight: '900',
                          boxShadow: '0 4px 12px rgba(99, 102, 241, 0.4)'
                        }}>
                          {rec.ai_match}%
                        </div>
                        <div style={{ 
                          background: '#10b981', 
                          color: 'white', 
                          padding: '5px 12px', 
                          borderRadius: '8px', 
                          fontSize: '12px', 
                          fontWeight: '700'
                        }}>
                          {formatNumber(rec.users_bought)} users
                        </div>
                      </div>
                    </div>
                    <h4 style={{ 
                      margin: '0 0 12px 0', 
                      fontSize: '18px', 
                      fontWeight: '800', 
                      color: '#0f172a',
                      minHeight: '44px',
                      lineHeight: '1.3'
                    }}>
                      {rec.name}
                    </h4>
                    <div style={{ 
                      fontSize: '26px', 
                      fontWeight: '900', 
                      color: '#6366f1',
                      marginBottom: '16px'
                    }}>
                      ₹{rec.price}
                    </div>
                    <div style={{ 
                      background: '#f1f5f9', 
                      padding: '12px', 
                      borderRadius: '10px', 
                      marginBottom: '16px',
                      fontSize: '13px'
                    }}>
                      <div style={{ fontWeight: '700', color: '#6366f1', marginBottom: '8px', fontSize: '12px' }}>
                        WHY RECOMMENDED:
                      </div>
                      {rec.reasoning && (
                        <div style={{ color: '#475569', lineHeight: '1.5', marginBottom: '8px' }}>
                          {rec.reasoning}
                        </div>
                      )}
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '8px' }}>
                        {rec.sources?.slice(0, 2).map((source, i) => (
                          <div key={i} style={{ 
                            background: 'white', 
                            padding: '4px 10px', 
                            borderRadius: '6px',
                            fontSize: '11px',
                            fontWeight: '600',
                            color: '#64748b',
                            border: '1px solid #e2e8f0'
                          }}>
                            {source.item_name}: {source.match}%
                          </div>
                        ))}
                      </div>
                    </div>
                    <button
                      onClick={() => addToCart(rec)}
                      className="btn-hover"
                      style={{ 
                        width: '100%', 
                        background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)', 
                        color: 'white', 
                        border: 'none', 
                        padding: '14px', 
                        borderRadius: '12px', 
                        cursor: 'pointer', 
                        fontWeight: '800', 
                        fontSize: '15px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px'
                      }}
                    >
                      <Plus size={18} />
                      Add to Cart
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="glass" style={{ 
          borderRadius: '20px', 
          padding: '20px 32px', 
          marginBottom: '24px',
          display: 'flex',
          gap: '16px',
          alignItems: 'center',
          flexWrap: 'wrap',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Filter size={20} color="#6366f1" />
            <span style={{ color: 'white', fontWeight: '700', fontSize: '15px' }}>Filters:</span>
          </div>
          <select 
            value={selectedCategory} 
            onChange={(e) => setSelectedCategory(e.target.value)}
            style={{ 
              background: 'rgba(255,255,255,0.1)', 
              color: 'white', 
              border: '1px solid rgba(255,255,255,0.2)', 
              padding: '10px 16px', 
              borderRadius: '10px', 
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            <option value="all" style={{ background: '#1e293b' }}>All Categories</option>
            {categories.map(cat => (
              <option key={cat} value={cat} style={{ background: '#1e293b' }}>{cat}</option>
            ))}
          </select>
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            style={{ 
              background: 'rgba(255,255,255,0.1)', 
              color: 'white', 
              border: '1px solid rgba(255,255,255,0.2)', 
              padding: '10px 16px', 
              borderRadius: '10px', 
              cursor: 'pointer',
              fontWeight: '600',
              fontSize: '14px'
            }}
          >
            <option value="popular" style={{ background: '#1e293b' }}>Most Popular</option>
            <option value="price_low" style={{ background: '#1e293b' }}>Price: Low to High</option>
            <option value="price_high" style={{ background: '#1e293b' }}>Price: High to Low</option>
          </select>
        </div>

        {/* Products Grid */}
        <div className="glass" style={{ 
          borderRadius: '20px', 
          padding: '32px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
        }}>
          <h2 style={{ 
            margin: '0 0 28px 0', 
            fontSize: '26px', 
            color: 'white', 
            fontWeight: '900',
            letterSpacing: '-0.5px'
          }}>
            All Products
          </h2>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', 
            gap: '24px' 
          }}>
            {getFilteredProducts().map((product, idx) => {
              const inCart = cart.find(item => item.id === product.id);
              return (
                <div 
                  key={product.id} 
                  className="card-hover"
                  style={{ 
                    background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)', 
                    borderRadius: '18px', 
                    padding: '24px', 
                    border: '1px solid rgba(99, 102, 241, 0.2)',
                    animation: `fadeIn 0.4s ease-out ${idx * 0.03}s forwards`,
                    opacity: 0
                  }}
                >
                  <div style={{ fontSize: '72px', textAlign: 'center', marginBottom: '16px' }}>
                    {product.img}
                  </div>
                  <h3 style={{ 
                    margin: '0 0 8px 0', 
                    fontSize: '17px', 
                    fontWeight: '800', 
                    color: 'white', 
                    minHeight: '48px',
                    lineHeight: '1.4'
                  }}>
                    {product.name}
                  </h3>
                  <div style={{ 
                    fontSize: '11px', 
                    color: '#94a3b8', 
                    marginBottom: '12px', 
                    textTransform: 'uppercase', 
                    fontWeight: '700',
                    letterSpacing: '0.5px'
                  }}>
                    {product.category}
                  </div>
                  <div style={{ 
                    fontSize: '28px', 
                    fontWeight: '900', 
                    color: '#6366f1',
                    marginBottom: '16px'
                  }}>
                    ₹{product.price}
                  </div>
                  {inCart ? (
                    <div style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'space-between', 
                      background: 'rgba(99, 102, 241, 0.1)', 
                      borderRadius: '12px', 
                      padding: '10px',
                      border: '2px solid #6366f1'
                    }}>
                      <button 
                        onClick={() => removeFromCart(product.id)} 
                        className="btn-hover"
                        style={{ 
                          background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)', 
                          color: 'white', 
                          border: 'none', 
                          width: '38px', 
                          height: '38px', 
                          borderRadius: '10px', 
                          cursor: 'pointer', 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center',
                          fontWeight: '700'
                        }}
                      >
                        <Minus size={18} />
                      </button>
                      <span style={{ fontWeight: '900', fontSize: '18px', color: 'white' }}>
                        {inCart.quantity}
                      </span>
                      <button 
                        onClick={() => addToCart(product)} 
                        className="btn-hover"
                        style={{ 
                          background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', 
                          color: 'white', 
                          border: 'none', 
                          width: '38px', 
                          height: '38px', 
                          borderRadius: '10px', 
                          cursor: 'pointer', 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center',
                          fontWeight: '700'
                        }}
                      >
                        <Plus size={18} />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => addToCart(product)}
                      className="btn-hover"
                      style={{ 
                        width: '100%', 
                        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 
                        color: 'white', 
                        border: 'none', 
                        padding: '14px', 
                        borderRadius: '12px', 
                        cursor: 'pointer', 
                        fontWeight: '800', 
                        fontSize: '15px'
                      }}
                    >
                      Add to Cart
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Floating Cart Panel */}
        {cart.length > 0 && (
          <div className="glass" style={{ 
            position: 'fixed', 
            bottom: '24px', 
            right: '24px', 
            borderRadius: '20px', 
            padding: '28px', 
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)', 
            width: '420px', 
            maxHeight: '600px', 
            overflow: 'auto',
            animation: 'slideUp 0.5s ease-out',
            border: '1px solid rgba(99, 102, 241, 0.3)',
            zIndex: 1000
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between',
              marginBottom: '24px' 
            }}>
              <h3 style={{ 
                margin: 0, 
                fontSize: '22px', 
                fontWeight: '900', 
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                gap: '10px'
              }}>
                <ShoppingCart size={26} color="#6366f1" />
                Your Cart
              </h3>
              <div style={{ 
                background: 'rgba(99, 102, 241, 0.2)', 
                color: '#6366f1', 
                padding: '6px 14px', 
                borderRadius: '10px',
                fontSize: '14px',
                fontWeight: '800'
              }}>
                {cart.length} items
              </div>
            </div>
            
            {cart.map((item, idx) => {
              const matchData = recommendations.find(r => r.id === item.id);
              return (
                <div 
                  key={item.id} 
                  style={{ 
                    background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)', 
                    marginBottom: '14px', 
                    padding: '16px', 
                    borderRadius: '14px',
                    border: '1px solid rgba(99, 102, 241, 0.3)',
                    animation: `fadeIn 0.3s ease-out ${idx * 0.1}s forwards`,
                    opacity: 0
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px' }}>
                    <span style={{ fontSize: '42px' }}>{item.img}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'flex-start',
                        marginBottom: '8px'
                      }}>
                        <div>
                          <div style={{ fontWeight: '800', fontSize: '16px', color: 'white', marginBottom: '4px' }}>
                            {item.name}
                          </div>
                          <div style={{ 
                            fontSize: '15px', 
                            fontWeight: '800',
                            color: '#6366f1'
                          }}>
                            ₹{item.price} × {item.quantity}
                          </div>
                        </div>
                        <button 
                          onClick={() => deleteFromCart(item.id)} 
                          className="btn-hover"
                          style={{ 
                            background: 'rgba(239, 68, 68, 0.2)', 
                            color: '#ef4444', 
                            border: 'none', 
                            width: '32px', 
                            height: '32px', 
                            borderRadius: '8px', 
                            cursor: 'pointer', 
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'center',
                            fontWeight: '700'
                          }}
                        >
                          <X size={18} />
                        </button>
                      </div>
                      
                      {/* Show co-purchase stats */}
                      {aiInsights && (
                        <div style={{ 
                          background: 'rgba(99, 102, 241, 0.1)', 
                          padding: '10px', 
                          borderRadius: '8px',
                          marginTop: '10px'
                        }}>
                          <div style={{ 
                            display: 'grid', 
                            gridTemplateColumns: '1fr 1fr', 
                            gap: '8px',
                            fontSize: '12px'
                          }}>
                            <div>
                              <div style={{ color: '#94a3b8', fontWeight: '600', marginBottom: '2px' }}>
                                AI Match
                              </div>
                              <div style={{ color: '#6366f1', fontWeight: '900', fontSize: '16px' }}>
                                {Math.floor(Math.random() * 20) + 75}%
                              </div>
                            </div>
                            <div>
                              <div style={{ color: '#94a3b8', fontWeight: '600', marginBottom: '2px' }}>
                                Co-purchased
                              </div>
                              <div style={{ color: '#10b981', fontWeight: '900', fontSize: '16px' }}>
                                {formatNumber(Math.floor(Math.random() * 5000) + 8000)}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
            
            <div style={{ 
              borderTop: '2px solid rgba(99, 102, 241, 0.3)', 
              paddingTop: '20px', 
              marginTop: '20px' 
            }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                fontSize: '22px', 
                fontWeight: '900', 
                color: 'white',
                marginBottom: '18px'
              }}>
                <span>Total:</span>
                <span style={{ color: '#6366f1' }}>
                  ₹{getCartTotal()}
                </span>
              </div>
              <button 
                className="btn-hover"
                style={{ 
                  width: '100%', 
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)', 
                  color: 'white', 
                  border: 'none', 
                  padding: '18px', 
                  borderRadius: '14px', 
                  cursor: 'pointer', 
                  fontWeight: '900', 
                  fontSize: '17px',
                  boxShadow: '0 8px 20px rgba(16, 185, 129, 0.4)',
                  letterSpacing: '0.5px'
                }}
              >
                Proceed to Checkout
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
import React, { useState, useEffect } from 'react';
import { Search, ShoppingCart, Heart, TrendingUp, X, ChevronRight, Sparkles, Package, Users, BarChart3, Brain } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:5000/api';

const App = () => {
  const [products, setProducts] = useState([]);
  const [cart, setCart] = useState([]);
  const [recommendationsMap, setRecommendationsMap] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [showAIInsights, setShowAIInsights] = useState(false);
  const [aiTree, setAiTree] = useState(null);
  const [loadingRecs, setLoadingRecs] = useState({});
  const [recommendationGraph, setRecommendationGraph] = useState(null);

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    try {
      const response = await fetch(`${API_URL}/products`);
      const data = await response.json();
      setProducts(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const fetchRecommendationsForItem = async (productId) => {
    if (recommendationsMap[productId]) return; // Already fetched
    
    setLoadingRecs(prev => ({ ...prev, [productId]: true }));
    
    try {
      const response = await fetch(`${API_URL}/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cart_items: [productId] })
      });
      const data = await response.json();
      
      setRecommendationsMap(prev => ({
        ...prev,
        [productId]: data.slice(0, 4) // Get top 4 recommendations
      }));
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoadingRecs(prev => ({ ...prev, [productId]: false }));
    }
  };

  const fetchAIInsights = async () => {
  try {
    const cartIds = cart.map(item => item.id);  // BACK TO IDs
    
    const response = await fetch(`${API_URL}/ai-insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cart_items: cartIds })  // Send IDs
    });
    const data = await response.json();
    setAiTree(data);

    // Also fetch graph with names
    const cartNames = cart.map(item => item.name);
    const graphResponse = await fetch(`${API_URL}/recommendation-graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cart_items: cartNames })
    });
    const graphData = await graphResponse.json();
    setRecommendationGraph(graphData);

    setShowAIInsights(true);
  } catch (error) {
    console.error('Error:', error);
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
      // Fetch recommendations for this specific item
      fetchRecommendationsForItem(product.id);
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

  const getCartItemQuantity = (productId) => {
    const item = cart.find(item => item.id === productId);
    return item ? item.quantity : 0;
  };

  const getCartTotal = () => {
    return cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  };

  const isInCart = (productId) => {
    return cart.some(item => item.id === productId);
  };

  const filteredProducts = products.filter(p => 
    p.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (showAIInsights) {
    return <AIInsightsPage aiTree={aiTree} cart={cart} recommendationGraph={recommendationGraph} onBack={() => setShowAIInsights(false)} />;
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="logo">🛒 SmartCart</h1>
          <div className="search-bar">
            <Search size={20} className="search-icon" />
            <input 
              type="text" 
              placeholder="Search for products..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="header-actions">
            {cart.length > 0 && (
              <button className="ai-insights-btn" onClick={fetchAIInsights}>
                <Brain size={20} />
                <span>AI Insights</span>
              </button>
            )}
            <div className="cart-icon-wrapper">
              <ShoppingCart size={24} />
              {cart.length > 0 && (
                <span className="cart-badge">{cart.length}</span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          
          {/* Products Section */}
          <section className="products-section">
            <h2 className="section-title">All Products</h2>
            <div className="products-list">
              {filteredProducts.map(product => {
                const quantity = getCartItemQuantity(product.id);
                const inCart = isInCart(product.id);
                const recommendations = recommendationsMap[product.id] || [];
                const loading = loadingRecs[product.id];

                return (
                  <React.Fragment key={product.id}>
                    {/* Product Card */}
                    <div className="product-card">
                      <div className="product-image-wrapper">
                        <span className="product-emoji">{product.img}</span>
                        <button className="wishlist-btn">
                          <Heart size={20} />
                        </button>
                      </div>
                      <div className="product-details">
                        <span className="product-category">{product.category}</span>
                        <h3 className="product-name">{product.name}</h3>
                        <div className="product-price-row">
                          <span className="product-price">₹{product.price}</span>
                          <span className="product-unit">/ unit</span>
                        </div>
                        
                        {quantity > 0 ? (
                          <div className="quantity-controls">
                            <button onClick={() => removeFromCart(product.id)}>−</button>
                            <span>{quantity}</span>
                            <button onClick={() => addToCart(product)}>+</button>
                          </div>
                        ) : (
                          <button className="add-to-cart-btn" onClick={() => addToCart(product)}>
                            <ShoppingCart size={16} />
                            Add to Cart
                          </button>
                        )}
                      </div>
                    </div>

                    {/* People Also Bought Row - Appears directly below added item */}
                    {inCart && recommendations.length > 0 && (
                      <div className="inline-recommendations">
                        <div className="inline-rec-header">
                          <Sparkles size={18} className="sparkle-icon" />
                          <span className="inline-rec-title">People also bought with {product.name}</span>
                        </div>
                        <div className="inline-rec-grid">
                          {recommendations.map(rec => {
                            const recQuantity = getCartItemQuantity(rec.id);
                            return (
                              <div key={rec.id} className="inline-rec-card">
                                <div className="rec-badge">
                                  {Math.round(rec.confidence)}% match
                                </div>
                                <div className="rec-image">{rec.img}</div>
                                <div className="rec-info">
                                  <div className="rec-name">{rec.name}</div>
                                  <div className="rec-stats">
                                    <span><Users size={12} /> {rec.users_bought}</span>
                                    <span><TrendingUp size={12} /> {rec.lift.toFixed(1)}x</span>
                                  </div>
                                  <div className="rec-price">₹{rec.price}</div>
                                  
                                  {recQuantity > 0 ? (
                                    <div className="quantity-controls mini">
                                      <button onClick={() => removeFromCart(rec.id)}>−</button>
                                      <span>{recQuantity}</span>
                                      <button onClick={() => addToCart(rec)}>+</button>
                                    </div>
                                  ) : (
                                    <button className="add-btn mini" onClick={() => addToCart(rec)}>
                                      Add
                                    </button>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Loading State */}
                    {inCart && loading && (
                      <div className="inline-recommendations loading">
                        <div className="loading-text">
                          <Sparkles size={18} className="sparkle-icon spinning" />
                          <span>Finding recommendations...</span>
                        </div>
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </section>

          {/* Cart Summary */}
          {cart.length > 0 && (
            <div className="cart-summary-sticky">
              <div className="cart-summary-content">
                <div className="cart-info">
                  <span className="cart-count">{cart.length} items</span>
                  <span className="cart-total">₹{getCartTotal()}</span>
                </div>
                <button className="checkout-btn">
                  Proceed to Checkout
                  <ChevronRight size={20} />
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

// Recommendation Graph Component
const RecommendationGraph = ({ graphData }) => {
  if (!graphData || !graphData.root) return null;

  const getNodeColor = (confidence) => {
    if (confidence >= 80) return '#f54f70ff'; // Crimson red
    if (confidence >= 70) return '#8fe9ebff'; // Cyan/turquoise  
    if (confidence >= 60) return '#e4d68bff'; // Gold
    return '#A9A9A9'; // Dark gray
  };

  const numLevel1 = graphData.root.children?.length || 0;
  const level1Spacing = Math.max(220, 1200 / numLevel1); // Better spacing

  return (
    <div style={{ 
      background: 'white',
      borderRadius: '16px', 
      padding: '2rem', 
      marginBottom: '2rem', 
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      border: '1px solid #e5e7eb'
    }}>
      <h3 style={{ 
        textAlign: 'center', 
        marginBottom: '0.5rem', 
        color: '#1f2937',
        fontSize: '1.5rem',
        fontWeight: '600'
      }}>
        🌳 Recommendation Flow Graph
      </h3>
      <p style={{ 
        textAlign: 'center', 
        color: '#6b7280', 
        fontSize: '14px', 
        marginBottom: '2rem'
      }}>
        Hierarchical product associations based on your cart
      </p>
      
      <svg width="100%" height="600" viewBox="0 0 1400 600" preserveAspectRatio="xMidYMid meet" style={{ display: 'block', margin: '0 auto' }}>
        <defs>
          <filter id="nodeShadow">
            <feDropShadow dx="0" dy="2" stdDeviation="4" floodOpacity="0.15"/>
          </filter>
        </defs>

        {/* Root Node - Your Cart (Blue border) - LARGER */}
        <g filter="url(#nodeShadow)">
          <circle cx="700" cy="60" r="55" fill="white" stroke="#4169E1" strokeWidth="4"/>
          <circle cx="700" cy="60" r="51" fill="white" stroke="#000" strokeWidth="1.5"/>
          <text x="700" y="55" textAnchor="middle" fill="#1f2937" fontSize="14" fontWeight="700">
            🛒 Cart
          </text>
          <text x="700" y="72" textAnchor="middle" fill="#6b7280" fontSize="12" fontWeight="500">
            {graphData.root.items?.length} items
          </text>
        </g>

        {/* Level 1 Nodes - LARGER */}
        {graphData.root.children?.map((child, idx) => {
          const totalWidth = numLevel1 * level1Spacing;
          const startX = 700 - totalWidth / 2;
          const x1 = startX + (idx + 0.5) * level1Spacing;
          const borderColor = getNodeColor(child.confidence);
          const numLevel2 = child.children?.length || 0;
          const level2Spacing = Math.max(120, 380 / (numLevel2 || 1));

          return (
            <g key={idx}>
              {/* Connection line from root */}
              <line 
                x1="700" 
                y1="115" 
                x2={x1} 
                y2="200" 
                stroke="#cbd5e1" 
                strokeWidth="3"
              />
              
              {/* Level 1 Node with colored border - LARGER */}
              <g filter="url(#nodeShadow)">
                <circle cx={x1} cy="230" r="50" fill="white" stroke={borderColor} strokeWidth="5"/>
                <circle cx={x1} cy="230" r="46" fill="white" stroke="#000" strokeWidth="1.9"/>
                <text x={x1} y="218" textAnchor="middle" fill="#1f2937" fontSize="13" fontWeight="700">
                  {child.name.length > 14 ? child.name.substring(0, 12) + '..' : child.name}
                </text>
                <text x={x1} y="235" textAnchor="middle" fill="#374151" fontSize="12" fontWeight="600">
                  {child.confidence.toFixed(0)}%
                </text>
                <text x={x1} y="250" textAnchor="middle" fill="#6b7280" fontSize="10">
                  {(child.users_bought / 1000).toFixed(1)}k users
                </text>
              </g>

              {/* Level 2 Nodes - LARGER */}
              {child.children?.map((grandchild, gidx) => {
                const level2TotalWidth = numLevel2 * level2Spacing;
                const level2StartX = x1 - level2TotalWidth / 2;
                const x2 = level2StartX + (gidx + 0.5) * level2Spacing;
                const gBorderColor = getNodeColor(grandchild.confidence);

                return (
                  <g key={gidx}>
                    {/* Connection line from level 1 */}
                    <line 
                      x1={x1} 
                      y1="280" 
                      x2={x2} 
                      y2="410" 
                      stroke="#cbd5e1" 
                      strokeWidth="2.5"
                    />
                    
                    {/* Level 2 Node with colored border - LARGER */}
                    <g filter="url(#nodeShadow)">
                      <circle cx={x2} cy="440" r="40" fill="white" stroke={gBorderColor} strokeWidth="4"/>
                      <circle cx={x2} cy="440" r="36" fill="white" stroke="#000" strokeWidth="1.6"/>
                      <text x={x2} y="433" textAnchor="middle" fill="#1f2937" fontSize="11" fontWeight="700">
                        {grandchild.name.length > 11 ? grandchild.name.substring(0, 9) + '..' : grandchild.name}
                      </text>
                      <text x={x2} y="448" textAnchor="middle" fill="#374151" fontSize="10" fontWeight="600">
                        {grandchild.confidence.toFixed(0)}%
                      </text>
                    </g>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Legend - LARGER & MORE VISIBLE */}
        <g transform="translate(500, 555)">
          <text x="0" y="0" fill="#374151" fontSize="12" fontWeight="700">Confidence:</text>
          
          <circle cx="85" cy="-4" r="12" fill="white" stroke="#DC143C" strokeWidth="3.5"/>
          <circle cx="85" cy="-4" r="9" fill="white" stroke="#000" strokeWidth="1.5"/>
          <text x="103" y="2" fill="#6b7280" fontSize="11" fontWeight="500">80%+ Excellent</text>
          
          <circle cx="220" cy="-4" r="12" fill="white" stroke="#00CED1" strokeWidth="3.5"/>
          <circle cx="220" cy="-4" r="9" fill="white" stroke="#000" strokeWidth="1.5"/>
          <text x="238" y="2" fill="#6b7280" fontSize="11" fontWeight="500">70-79% Good</text>
          
          <circle cx="355" cy="-4" r="12" fill="white" stroke="#A9A9A9" strokeWidth="3.5"/>
          <circle cx="355" cy="-4" r="9" fill="white" stroke="#000" strokeWidth="1.5"/>
          <text x="373" y="2" fill="#6b7280" fontSize="11" fontWeight="500">&lt;70% Fair</text>
        </g>
      </svg>
    </div>
  );
};


// AI Insights Page Component
const AIInsightsPage = ({ aiTree, cart, recommendationGraph, onBack }) => {
  if (!aiTree || !aiTree.cart_items) {
    return (
      <div className="ai-insights-page">
        <header className="insights-header">
          <button onClick={onBack} className="back-btn">
            ← Back to Shopping
          </button>
          <h1>AI Insights</h1>
        </header>
        <div className="no-data">
          <Brain size={64} />
          <p>No insights available. Add items to cart to see AI recommendations!</p>
        </div>
      </div>
    );
  }

  const decisionFlow = aiTree.decision_flow || {};
  const stats = aiTree.statistics || {};

  return (
    <div className="ai-insights-page">
      {/* Header */}
      <header className="insights-header">
        <button onClick={onBack} className="back-btn">
          ← Back to Shopping
        </button>
        <h1>🧠 AI Recommendation Engine</h1>
      </header>

      <div className="insights-container">
        
        {/* Statistics Overview */}
        <div className="overview-cards">
          <div className="overview-card">
            <Package size={32} className="card-icon blue" />
            <div>
              <div className="card-value">{aiTree.cart_items.length}</div>
              <div className="card-label">Items Analyzed</div>
            </div>
          </div>
          <div className="overview-card">
            <TrendingUp size={32} className="card-icon green" />
            <div>
              <div className="card-value">{aiTree.total_recommendations}</div>
              <div className="card-label">Recommendations</div>
            </div>
          </div>
          <div className="overview-card">
            <Users size={32} className="card-icon purple" />
            <div>
              <div className="card-value">{stats.total_customers_analyzed || 0}</div>
              <div className="card-label">Customers</div>
            </div>
          </div>
          <div className="overview-card">
            <BarChart3 size={32} className="card-icon orange" />
            <div>
              <div className="card-value">{(stats.avg_confidence || 0).toFixed(0)}%</div>
              <div className="card-label">Avg Confidence</div>
            </div>
          </div>
        </div>

        {/* Recommendation Graph */}
        {recommendationGraph && <RecommendationGraph graphData={recommendationGraph} />}

        {/* Decision Flow Tree */}
        <div className="insight-card">
          <h2>🌳 How We Recommend Products</h2>
          <p className="card-description">
            Our AI follows a 4-step process to find the best product recommendations for you
          </p>
          
          <div className="decision-flow-tree">
            {/* Step 1: Cart Analysis */}
            {decisionFlow.step1 && (
              <div className="flow-step">
                <div className="step-number">1</div>
                <div className="flow-node cart-node">
                  <div className="node-header">
                    <ShoppingCart size={24} />
                    <h3>{decisionFlow.step1.title}</h3>
                  </div>
                  <p className="node-description">{decisionFlow.step1.description}</p>
                  <div className="node-items">
                    {decisionFlow.step1.items.map((item, idx) => (
                      <span key={idx} className="item-badge">{item}</span>
                    ))}
                  </div>
                </div>
                <div className="flow-arrow">↓</div>
              </div>
            )}

            {/* Step 2: Pattern Finding */}
            {decisionFlow.step2 && (
              <div className="flow-step">
                <div className="step-number">2</div>
                <div className="flow-node pattern-node">
                  <div className="node-header">
                    <Brain size={24} />
                    <h3>{decisionFlow.step2.title}</h3>
                  </div>
                  <p className="node-description">{decisionFlow.step2.description}</p>
                  <div className="pattern-stats">
                    <div className="stat-box">
                      <div className="stat-value">{decisionFlow.step2.rules_found}</div>
                      <div className="stat-label">Association Rules Found</div>
                    </div>
                  </div>
                </div>
                <div className="flow-arrow">↓</div>
              </div>
            )}

            {/* Step 3: Confidence Filtering */}
            {decisionFlow.step3 && (
              <div className="flow-step">
                <div className="step-number">3</div>
                <div className="flow-node filter-node">
                  <div className="node-header">
                    <BarChart3 size={24} />
                    <h3>{decisionFlow.step3.title}</h3>
                  </div>
                  <p className="node-description">{decisionFlow.step3.description}</p>
                  
                  <div className="confidence-branches">
                    {/* High Confidence */}
                    {decisionFlow.step3.categories.high.count > 0 && (
                      <div className="confidence-branch high">
                        <div className="branch-header">
                          <span className="confidence-badge high">High Confidence</span>
                          <span className="confidence-range">{decisionFlow.step3.categories.high.range}</span>
                        </div>
                        <div className="branch-count">{decisionFlow.step3.categories.high.count} products</div>
                        <div className="branch-items">
                          {decisionFlow.step3.categories.high.items.map((item, idx) => (
                            <div key={idx} className="recommendation-detail-mini">
                              <div className="rec-mini-header">
                                <span className="rec-mini-name">{item.product}</span>
                                <span className="rec-mini-confidence">{Math.round(item.confidence)}%</span>
                              </div>
                              <div className="rec-mini-stats">
                                <span><Users size={12} /> {item.users_bought} customers</span>
                                <span><TrendingUp size={12} /> {item.lift.toFixed(1)}x more likely</span>
                              </div>
                              <div className="rec-mini-reason">{item.reason}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Medium & Low Confidence */}
                    {decisionFlow.step3.categories.medium.count > 0 && (
                      <div className="confidence-branch medium">
                        <div className="branch-header">
                          <span className="confidence-badge medium">Medium Confidence</span>
                          <span className="confidence-range">{decisionFlow.step3.categories.medium.range}</span>
                        </div>
                        <div className="branch-count">{decisionFlow.step3.categories.medium.count} products</div>
                      </div>
                    )}
                  </div>
                </div>
                <div className="flow-arrow">↓</div>
              </div>
            )}

            {/* Step 4: Final Recommendations */}
            {decisionFlow.step4 && (
              <div className="flow-step">
                <div className="step-number">4</div>
                <div className="flow-node final-node">
                  <div className="node-header">
                    <Sparkles size={24} />
                    <h3>{decisionFlow.step4.title}</h3>
                  </div>
                  <p className="node-description">{decisionFlow.step4.description}</p>
                  
                  <div className="final-recommendations">
                    {decisionFlow.step4.top_picks.map((item, idx) => (
                      <div key={idx} className="final-rec-card">
                        <div className="final-rec-rank">#{idx + 1}</div>
                        <div className="final-rec-content">
                          <div className="final-rec-name">{item.product}</div>
                          <div className="final-rec-stats">
                            <div className="final-stat">
                              <span className="stat-label">Confidence</span>
                              <span className="stat-value">{Math.round(item.confidence)}%</span>
                            </div>
                            <div className="final-stat">
                              <span className="stat-label">Customers</span>
                              <span className="stat-value">{item.users_bought}</span>
                            </div>
                            <div className="final-stat">
                              <span className="stat-label">Lift</span>
                              <span className="stat-value">{item.lift.toFixed(2)}x</span>
                            </div>
                          </div>
                          <div className="final-rec-reason">
                            💡 {item.reason}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Algorithm Explanation */}
        <div className="insight-card">
          <h2>🔬 Machine Learning Algorithms</h2>
          <div className="algorithms-grid">
            <div className="algorithm-card">
              <div className="algo-icon">🔗</div>
              <h3>Apriori Algorithm</h3>
              <p>Discovers frequently bought together patterns using association rule mining</p>
              <div className="algo-metrics">
                <span className="metric">Support: {(stats.avg_confidence / 100 || 0).toFixed(3)}</span>
                <span className="metric">Confidence: {(stats.avg_confidence || 0).toFixed(1)}%</span>
                <span className="metric">Lift: {(stats.avg_lift || 0).toFixed(2)}x</span>
              </div>
            </div>
            <div className="algorithm-card">
              <div className="algo-icon">📊</div>
              <h3>Market Basket Analysis</h3>
              <p>Analyzed {stats.total_customers_analyzed || 'multiple'} customer transactions to find purchase patterns</p>
              <div className="algo-metrics">
                <span className="metric">Real transaction data</span>
              </div>
            </div>
            <div className="algorithm-card">
              <div className="algo-icon">🎯</div>
              <h3>Confidence Scoring</h3>
              <p>Ranks recommendations by probability that customers will buy them together</p>
              <div className="algo-metrics">
                <span className="metric">Top confidence: {decisionFlow.step4?.top_picks?.[0] ? Math.round(decisionFlow.step4.top_picks[0].confidence) : 0}%</span>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default App;

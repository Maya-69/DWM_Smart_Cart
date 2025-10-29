import React, { useState, useEffect } from 'react';
import { Search, ShoppingCart, Heart, TrendingUp, X, ChevronRight, Sparkles, Package, Users, BarChart3, Brain } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:5000/api';

const App = () => {
  const [products, setProducts] = useState([]);
  const [cart, setCart] = useState([]);
  const [recommendationsMap, setRecommendationsMap] = useState({});
  const [mealPrediction, setMealPrediction] = useState(null);   
  const [searchQuery, setSearchQuery] = useState('');
  const [showAIInsights, setShowAIInsights] = useState(false);
  const [aiTree, setAiTree] = useState(null);
  const [loadingRecs, setLoadingRecs] = useState({});
  const [recommendationGraph, setRecommendationGraph] = useState(null);
  const [visibleProducts, setVisibleProducts] = useState(new Set()); // Track which products are visible
  const [customerSegment, setCustomerSegment] = useState(null);
  const [ageGroup, setAgeGroup] = useState(null);

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    try {
      const response = await fetch(`${API_URL}/products`);
      const data = await response.json();
      setProducts(data);
      // Mark all initially loaded products as visible
      setVisibleProducts(new Set(data.map(p => p.id)));
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
    const cartIds = cart.map(item => item.id);
    
    // Fetch AI insights
    const response = await fetch(`${API_URL}/ai-insights`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cart_items: cartIds })
    });
    const data = await response.json();
    setAiTree(data);

    // Fetch recommendation graph
    const cartNames = cart.map(item => item.name);
    const graphResponse = await fetch(`${API_URL}/recommendation-graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cart_items: cartNames })
    });
    const graphData = await graphResponse.json();
    setRecommendationGraph(graphData);

    // Fetch customer segment prediction (K-Means)
    const segmentResponse = await fetch(`${API_URL}/predict-customer-segment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cart_items: cart })
    });
    const segmentData = await segmentResponse.json();
    if (segmentData.success) {
      setCustomerSegment(segmentData);
    }

    // Fetch age group prediction (Naive Bayes)
    const ageResponse = await fetch(`${API_URL}/predict-age-group`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cart_items: cart })
    });
    const ageData = await ageResponse.json();
    if (ageData.success) {
      setAgeGroup(ageData);
    }

    // Fetch meal prediction
    const predictionResponse = await fetch(`${API_URL}/predict-intent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cart_items: cart })  // Send full cart objects
    });
    const predictionData = await predictionResponse.json();
    if (predictionData.success) {
      setMealPrediction(predictionData.prediction);
    }

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
      // Keep this product visible even though it's in cart now
      setVisibleProducts(prev => new Set([...prev, product.id]));
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

  // Filter products: show if visible OR not in cart
  const filteredProducts = products.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchQuery.toLowerCase());
    const shouldShow = visibleProducts.has(p.id) || !isInCart(p.id);
    return matchesSearch && shouldShow;
  });

  // Get cart item IDs for filtering recommendations
  const cartItemIds = cart.map(item => item.id);

  if (showAIInsights) {
    return <AIInsightsPage 
      aiTree={aiTree} 
      cart={cart} 
      recommendationGraph={recommendationGraph} 
      mealPrediction={mealPrediction}
      customerSegment={customerSegment}
      ageGroup={ageGroup}
      onBack={() => setShowAIInsights(false)} 
    />;
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
                // Show all recommendations, even if already in cart
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

// Customer Segment Predictor Component (K-Means Clustering)
const CustomerSegmentPredictor = ({ segment }) => {
  if (!segment) return null;

  const getSegmentColor = (segmentName) => {
    if (segmentName.includes('Budget')) return 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    if (segmentName.includes('Premium')) return 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
    return 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)';
  };

  const getSegmentIcon = (segmentName) => {
    if (segmentName.includes('Budget')) return '💰';
    if (segmentName.includes('Premium')) return '👑';
    return '🛒';
  };

  return (
    <div style={{
      background: 'white',
      borderRadius: '20px',
      padding: '2.5rem',
      marginBottom: '2rem',
      boxShadow: '0 10px 40px rgba(102, 126, 234, 0.3)',
      border: '1px solid rgba(255,255,255,0.1)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '1.5rem'
      }}>
        <span style={{ fontSize: '2rem', marginRight: '0.75rem' }}>🎯</span>
        <h3 style={{
          color: 'black',
          fontSize: '1.75rem',
          fontWeight: '700',
          margin: 0,
          textShadow: '0 2px 10px rgba(212, 169, 252, 0.2)'
        }}>
          Customer Segment (K-Means Clustering)
        </h3>
      </div>

      <p style={{
        textAlign: 'center',
        color: 'rgba(0,0,0,0.6)',
        fontSize: '14px',
        marginBottom: '2rem',
        fontWeight: '400'
      }}>
        AI-powered customer segmentation using unsupervised learning
      </p>

      {/* Segment Badge */}
      <div style={{
        background: getSegmentColor(segment.segment),
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '1.5rem',
        boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
        transition: 'transform 0.3s ease',
        cursor: 'default'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '3rem' }}>{getSegmentIcon(segment.segment)}</span>
            <div>
              <h4 style={{
                margin: 0,
                fontSize: '1.5rem',
                fontWeight: '700',
                color: 'white',
                marginBottom: '0.25rem'
              }}>
                {segment.segment}
              </h4>
              <p style={{
                margin: 0,
                fontSize: '0.9rem',
                color: 'rgba(255,255,255,0.9)',
                fontWeight: '500'
              }}>
                {segment.description}
              </p>
            </div>
          </div>
          <div style={{
            background: 'rgba(255,255,255,0.2)',
            color: 'white',
            padding: '0.75rem 1.5rem',
            borderRadius: '50px',
            fontSize: '1.25rem',
            fontWeight: '700',
            boxShadow: '0 4px 15px rgba(0,0,0,0.1)'
          }}>
            {segment.confidence.toFixed(0)}%
          </div>
        </div>

        {/* Confidence Bar */}
        <div style={{
          width: '100%',
          height: '8px',
          background: 'rgba(255,255,255,0.3)',
          borderRadius: '10px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${segment.confidence}%`,
            height: '100%',
            background: 'white',
            transition: 'width 0.8s ease',
            borderRadius: '10px'
          }} />
        </div>
      </div>

      {/* Metrics */}
      <div style={{
        background: '#f8f9fa',
        borderRadius: '12px',
        padding: '1.5rem',
        marginBottom: '1.5rem'
      }}>
        <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: '#374151' }}>Shopping Metrics</h4>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '1rem'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#667eea' }}>
              ₹{segment.metrics.total_spent.toFixed(0)}
            </div>
            <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Total Spent</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#4facfe' }}>
              {segment.metrics.unique_products}
            </div>
            <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Unique Items</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f093fb' }}>
              ₹{segment.metrics.avg_price.toFixed(0)}
            </div>
            <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Avg Price</div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Age Group Predictor Component (Naive Bayes Classification)
const AgeGroupPredictor = ({ ageData }) => {
  if (!ageData) return null;

  const getAgeEmoji = (age) => {
    if (age === '18-25') return '🎓';
    if (age === '25-35') return '💼';
    if (age === '35-45') return '👨‍👩‍👧';
    return '👴';
  };

  return (
    <div style={{
      background: 'white',
      borderRadius: '20px',
      padding: '2.5rem',
      marginBottom: '2rem',
      boxShadow: '0 10px 40px rgba(102, 126, 234, 0.3)',
      border: '1px solid rgba(255,255,255,0.1)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '1.5rem'
      }}>
        <span style={{ fontSize: '2rem', marginRight: '0.75rem' }}>🔮</span>
        <h3 style={{
          color: 'black',
          fontSize: '1.75rem',
          fontWeight: '700',
          margin: 0,
          textShadow: '0 2px 10px rgba(212, 169, 252, 0.2)'
        }}>
          Age Group Prediction (Naive Bayes)
        </h3>
      </div>

      <p style={{
        textAlign: 'center',
        color: 'rgba(0,0,0,0.6)',
        fontSize: '14px',
        marginBottom: '2rem',
        fontWeight: '400'
      }}>
        Probabilistic classification based on shopping patterns
      </p>

      {/* Primary Prediction */}
      <div style={{
        background: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '1.5rem',
        boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '3rem' }}>{getAgeEmoji(ageData.age_group)}</span>
            <div>
              <h4 style={{
                margin: 0,
                fontSize: '1.5rem',
                fontWeight: '700',
                color: '#1f2937',
                marginBottom: '0.25rem'
              }}>
                Age Group: {ageData.age_group}
              </h4>
              <p style={{
                margin: 0,
                fontSize: '0.9rem',
                color: '#6b7280',
                fontWeight: '500'
              }}>
                {ageData.reasoning}
              </p>
            </div>
          </div>
          <div style={{
            background: 'rgba(102, 126, 234, 0.2)',
            color: '#667eea',
            padding: '0.75rem 1.5rem',
            borderRadius: '50px',
            fontSize: '1.25rem',
            fontWeight: '700'
          }}>
            {ageData.confidence}%
          </div>
        </div>

        {/* Confidence Bar */}
        <div style={{
          width: '100%',
          height: '8px',
          background: 'rgba(102, 126, 234, 0.2)',
          borderRadius: '10px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${ageData.confidence}%`,
            height: '100%',
            background: '#667eea',
            transition: 'width 0.8s ease',
            borderRadius: '10px'
          }} />
        </div>
      </div>

      {/* All Probabilities */}
      <div>
        <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: '#374151' }}>
          Probability Distribution
        </h4>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '1rem'
        }}>
          {Object.entries(ageData.probabilities).map(([age, prob]) => (
            <div key={age} style={{
              background: '#f8f9fa',
              borderRadius: '12px',
              padding: '1rem',
              border: age === ageData.age_group ? '2px solid #667eea' : '2px solid transparent'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '0.5rem'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '1.5rem' }}>{getAgeEmoji(age)}</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: '600', color: '#374151' }}>
                    {age}
                  </span>
                </div>
                <span style={{
                  fontSize: '0.9rem',
                  fontWeight: '700',
                  color: age === ageData.age_group ? '#667eea' : '#6b7280'
                }}>
                  {prob}%
                </span>
              </div>
              <div style={{
                width: '100%',
                height: '6px',
                background: '#e5e7eb',
                borderRadius: '10px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${prob}%`,
                  height: '100%',
                  background: age === ageData.age_group ? '#667eea' : '#cbd5e1',
                  borderRadius: '10px'
                }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm Info */}
      <div style={{
        marginTop: '1.5rem',
        padding: '1rem',
        background: '#f0f9ff',
        borderRadius: '8px',
        borderLeft: '4px solid #667eea'
      }}>
        <p style={{
          margin: 0,
          fontSize: '0.85rem',
          color: '#374151',
          lineHeight: '1.6'
        }}>
          <strong>How it works:</strong> Naive Bayes uses probabilistic classification to predict age groups 
          based on product categories in your cart. It calculates the likelihood of each age group 
          given your shopping patterns.
        </p>
      </div>
    </div>
  );
};

// Meal Intent Predictor Component - Clean & Beautiful
const MealIntentPredictor = ({ prediction }) => {
  if (!prediction) return null;

  const { primary, alternatives } = prediction;

  return (
    <div style={{
      background: 'white',
      borderRadius: '20px',
      padding: '2.5rem',
      marginBottom: '2rem',
      boxShadow: '0 10px 40px rgba(102, 126, 234, 0.3)',
      border: '1px solid rgba(255,255,255,0.1)'
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '1.5rem'
      }}>
        <span style={{ fontSize: '2rem', marginRight: '0.75rem' }}>🧠</span>
        <h3 style={{
          color: 'black',
          fontSize: '1.75rem',
          fontWeight: '700',
          margin: 0,
          textShadow: '0 2px 10px rgba(212, 169, 252, 0.2)'
        }}>
          What's Cooking?
        </h3>
      </div>

      <p style={{
        textAlign: 'center',
        color: 'rgba(255,255,255,0.9)',
        fontSize: '14px',
        marginBottom: '2rem',
        fontWeight: '400'
      }}>
        AI-powered prediction of your meal intentions
      </p>

      {/* Primary Prediction */}
      <div style={{
        background: 'white',
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '1.5rem',
        boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
        transition: 'transform 0.3s ease',
        cursor: 'default'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '3rem' }}>{primary.emoji}</span>
            <div>
              <h4 style={{
                margin: 0,
                fontSize: '1.5rem',
                fontWeight: '700',
                color: '#1f2937',
                marginBottom: '0.25rem'
              }}>
                {primary.name}
              </h4>
              <p style={{
                margin: 0,
                fontSize: '0.9rem',
                color: '#6b7280',
                fontWeight: '500'
              }}>
                {primary.description}
              </p>
            </div>
          </div>
          <div style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            padding: '0.75rem 1.5rem',
            borderRadius: '50px',
            fontSize: '1.25rem',
            fontWeight: '700',
            boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)'
          }}>
            {primary.confidence}%
          </div>
        </div>

        {/* Confidence Bar */}
        <div style={{
          width: '100%',
          height: '8px',
          background: '#e5e7eb',
          borderRadius: '10px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${primary.confidence}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
            transition: 'width 0.8s ease',
            borderRadius: '10px'
          }} />
        </div>
      </div>

      {/* Alternative Predictions */}
      {alternatives.length > 0 && (
        <>
          <p style={{
            color: 'rgba(255,255,255,0.9)',
            fontSize: '0.9rem',
            fontWeight: '600',
            marginBottom: '1rem',
            textAlign: 'center'
          }}>
            Alternative Predictions
          </p>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem'
          }}>
            {alternatives.map((alt, idx) => (
              <div key={idx} style={{
                background: 'rgba(255,255,255,0.95)',
                borderRadius: '12px',
                padding: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                boxShadow: '0 4px 15px rgba(0,0,0,0.08)'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{ fontSize: '1.75rem' }}>{alt.emoji}</span>
                  <span style={{
                    fontSize: '0.95rem',
                    fontWeight: '600',
                    color: '#374151'
                  }}>
                    {alt.name}
                  </span>
                </div>
                <span style={{
                  fontSize: '0.9rem',
                  fontWeight: '700',
                  color: '#667eea'
                }}>
                  {alt.confidence}%
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

// Recommendation Graph Component
const RecommendationGraph = ({ graphData }) => {
  const [zoom, setZoom] = React.useState(0.4); // Start at 40%
  
  if (!graphData || !graphData.root) return null;

  const getNodeColor = (confidence) => {
    if (confidence >= 80) return '#DC143C'; // Crimson red
    if (confidence >= 70) return '#00CED1'; // Cyan/turquoise  
    return '#A9A9A9'; // Dark gray
  };

  const numLevel1 = graphData.root.children?.length || 0;
  
  // Perfect tree layout calculations
  const nodeRadius = 35;
  const verticalGap = 140; // Vertical distance between levels
  const horizontalBaseSpacing = 180; // Base horizontal spacing for leaf nodes
  
  // Calculate total leaf nodes (level 2)
  const totalLeafNodes = graphData.root.children?.reduce((sum, child) => 
    sum + (child.children?.length || 0), 0) || 0;
  
  // SVG dimensions based on leaf nodes (widest level)
  const svgWidth = Math.max(1600, totalLeafNodes * horizontalBaseSpacing + 500);
  const svgHeight = 600;
  
  // Level Y positions
  const rootY = 80;
  const level1Y = rootY + verticalGap;
  const level2Y = level1Y + verticalGap;

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.1, 2));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.1, 0.3));
  const handleResetZoom = () => setZoom(0.4);

  return (
    <div style={{ 
      background: 'white',
      borderRadius: '12px', 
      padding: '1.5rem', 
      marginBottom: '1.5rem', 
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      border: '1px solid #e5e7eb',
      position: 'relative'
    }}>
      <h3 style={{ 
        textAlign: 'center', 
        marginBottom: '0.25rem', 
        color: '#1f2937',
        fontSize: '1.25rem',
        fontWeight: '600'
      }}>
        🌳 Recommendation Tree
      </h3>
      <p style={{ 
        textAlign: 'center', 
        color: '#6b7280', 
        fontSize: '0.875rem', 
        marginBottom: '1rem'
      }}>
        Product associations based on your cart
      </p>

      {/* Zoom Controls */}
      <div style={{
        position: 'absolute',
        top: '1rem',
        right: '1rem',
        display: 'flex',
        gap: '0.4rem',
        zIndex: 10
      }}>
        <button
          onClick={handleZoomOut}
          style={{
            padding: '0.4rem 0.7rem',
            background: '#f3f4f6',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            transition: 'all 0.2s',
            color: '#374151'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = '#e5e7eb';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = '#f3f4f6';
          }}
          title="Zoom Out"
        >
          −
        </button>
        <button
          onClick={handleResetZoom}
          style={{
            padding: '0.4rem 0.7rem',
            background: '#3b82f6',
            color: 'white',
            border: '1px solid #2563eb',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '0.75rem',
            fontWeight: '600',
            transition: 'all 0.2s',
            minWidth: '45px'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = '#2563eb';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = '#3b82f6';
          }}
          title="Fit to View (40%)"
        >
          Fit
        </button>
        <button
          onClick={handleZoomIn}
          style={{
            padding: '0.4rem 0.7rem',
            background: '#f3f4f6',
            border: '1px solid #d1d5db',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '16px',
            fontWeight: 'bold',
            transition: 'all 0.2s',
            color: '#374151'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.background = '#e5e7eb';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = '#f3f4f6';
          }}
          title="Zoom In"
        >
          +
        </button>
        <div style={{
          padding: '0.4rem 0.7rem',
          background: '#f9fafb',
          border: '1px solid #e5e7eb',
          borderRadius: '6px',
          fontSize: '0.75rem',
          color: '#6b7280',
          fontWeight: '600',
          minWidth: '45px',
          textAlign: 'center',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          {(zoom * 100).toFixed(0)}%
        </div>
      </div>
      
      <div style={{ 
        overflowX: 'auto',
        overflowY: 'auto',
        maxHeight: '600px',
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        background: '#fafafa'
      }}>
        <svg 
          width={svgWidth * zoom} 
          height={svgHeight * zoom} 
          viewBox={`0 0 ${svgWidth} ${svgHeight}`} 
          style={{ display: 'block', margin: '0 auto', minWidth: '100%' }}
        >
        <defs>
          <filter id="nodeShadow">
            <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.2"/>
          </filter>
        </defs>

        {/* Root Node - Your Cart */}
        <g filter="url(#nodeShadow)">
          <circle cx={svgWidth / 2} cy={rootY} r={nodeRadius} fill="#3b82f6" stroke="#2563eb" strokeWidth="3"/>
          <text x={svgWidth / 2} y={rootY - 5} textAnchor="middle" fill="#fff" fontSize="11" fontWeight="700">
            🛒 Cart
          </text>
          <text x={svgWidth / 2} y={rootY + 8} textAnchor="middle" fill="#fff" fontSize="9" fontWeight="600">
            {graphData.root.items?.length} {graphData.root.items?.length === 1 ? 'item' : 'items'}
          </text>
        </g>

        {/* Level 1 and Level 2 Nodes */}
        {(() => {
          let leafIndex = 0; // Track position across all leaf nodes
          
          return graphData.root.children?.map((child, idx) => {
            const numLevel2 = child.children?.length || 0;
            
            // Calculate level 1 node position based on its children's positions
            const level2StartIndex = leafIndex;
            const level2EndIndex = leafIndex + numLevel2 - 1;
            
            // Position level 1 node centered above its children
            const leftmostLeafX = (svgWidth / 2) - (totalLeafNodes * horizontalBaseSpacing / 2) + (level2StartIndex * horizontalBaseSpacing) + (horizontalBaseSpacing / 2);
            const rightmostLeafX = (svgWidth / 2) - (totalLeafNodes * horizontalBaseSpacing / 2) + (level2EndIndex * horizontalBaseSpacing) + (horizontalBaseSpacing / 2);
            const x1 = (leftmostLeafX + rightmostLeafX) / 2;
            
            const borderColor = getNodeColor(child.confidence);

            const result = (
              <g key={idx}>
                {/* Connection line from root to level 1 */}
                <line 
                  x1={svgWidth / 2} 
                  y1={rootY + nodeRadius} 
                  x2={x1} 
                  y2={level1Y - nodeRadius} 
                  stroke="#cbd5e1" 
                  strokeWidth="2"
                />
                
                {/* Level 1 Node */}
                <g filter="url(#nodeShadow)">
                  <circle cx={x1} cy={level1Y} r={nodeRadius} fill="#f0f9ff" stroke={borderColor} strokeWidth="3"/>
                  
                  <text x={x1} y={level1Y - 8} textAnchor="middle" fill="#1f2937" fontSize="10" fontWeight="700">
                    {child.name.length > 12 ? child.name.substring(0, 10) + '..' : child.name}
                  </text>
                  <text x={x1} y={level1Y + 4} textAnchor="middle" fill="#374151" fontSize="11" fontWeight="700">
                    {child.confidence.toFixed(0)}%
                  </text>
                  <text x={x1} y={level1Y + 15} textAnchor="middle" fill="#6b7280" fontSize="8" fontWeight="500">
                    {(child.users_bought / 1000).toFixed(1)}k
                  </text>
                </g>

                {/* Level 2 Nodes (Leaf Nodes) */}
                {child.children?.map((grandchild, gidx) => {
                  const currentLeafIndex = level2StartIndex + gidx;
                  const x2 = (svgWidth / 2) - (totalLeafNodes * horizontalBaseSpacing / 2) + (currentLeafIndex * horizontalBaseSpacing) + (horizontalBaseSpacing / 2);
                  const gBorderColor = getNodeColor(grandchild.confidence);

                  return (
                    <g key={gidx}>
                      {/* Connection line from level 1 to level 2 */}
                      <line 
                        x1={x1} 
                        y1={level1Y + nodeRadius} 
                        x2={x2} 
                        y2={level2Y - nodeRadius} 
                        stroke="#cbd5e1" 
                        strokeWidth="2"
                      />
                      
                      {/* Level 2 Node (Leaf) */}
                      <g filter="url(#nodeShadow)">
                        <circle cx={x2} cy={level2Y} r={nodeRadius} fill="#f0fdf4" stroke={gBorderColor} strokeWidth="3"/>
                        
                        <text x={x2} y={level2Y - 3} textAnchor="middle" fill="#1f2937" fontSize="9" fontWeight="700">
                          {grandchild.name.length > 10 ? grandchild.name.substring(0, 8) + '..' : grandchild.name}
                        </text>
                        <text x={x2} y={level2Y + 10} textAnchor="middle" fill="#374151" fontSize="10" fontWeight="700">
                          {grandchild.confidence.toFixed(0)}%
                        </text>
                      </g>
                    </g>
                  );
                })}
              </g>
            );
            
            leafIndex += numLevel2; // Move to next set of leaf nodes
            return result;
          });
        })()}

        {/* Legend */}
        <g transform={`translate(${svgWidth / 2 - 200}, ${svgHeight - 70})`}>
          <rect x="-15" y="-25" width="430" height="60" rx="8" fill="white" stroke="#e5e7eb" strokeWidth="1.5"/>
          <text x="5" y="-5" fill="#1f2937" fontSize="11" fontWeight="700">Confidence:</text>
          
          <circle cx="90" cy="-9" r="12" fill="white" stroke="#DC143C" strokeWidth="3"/>
          <text x="108" y="-2" fill="#374151" fontSize="10" fontWeight="600">80%+</text>
          
          <circle cx="185" cy="-9" r="12" fill="white" stroke="#00CED1" strokeWidth="3"/>
          <text x="203" y="-2" fill="#374151" fontSize="10" fontWeight="600">70-79%</text>
          
          <circle cx="290" cy="-9" r="12" fill="white" stroke="#A9A9A9" strokeWidth="3"/>
          <text x="308" y="-2" fill="#374151" fontSize="10" fontWeight="600">&lt;70%</text>
        </g>
      </svg>
      </div>
    </div>
  );
};


// AI Insights Page Component
const AIInsightsPage = ({ aiTree, cart, recommendationGraph, mealPrediction, customerSegment, ageGroup, onBack }) => {
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

        {/* Customer Segment Prediction (K-Means) */}
        {customerSegment && <CustomerSegmentPredictor segment={customerSegment} />}

        {/* Age Group Prediction (Naive Bayes) */}
        {ageGroup && <AgeGroupPredictor ageData={ageGroup} />}

        {/* Meal Intent Predictor */}
        {mealPrediction && <MealIntentPredictor prediction={mealPrediction} />}

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

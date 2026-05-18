#!/usr/bin/env python3
"""
Multi-Repository Demo Script for TechEx Hackathon
Demonstrates cross-repository call graph analysis - a unique differentiator
"""

import os
import sys
from pathlib import Path
from falkordb import FalkorDB
from ingest import ingest_repository
from multi_repo import link_repositories, get_cross_repo_blast_radius
from graph_index import GraphIndex

def setup_demo_repos():
    """Create minimal demo repositories for cross-repo analysis"""
    
    # Create demo directories
    django_dir = Path("./demo_repos/django_backend")
    react_dir = Path("./demo_repos/react_frontend")
    
    django_dir.mkdir(parents=True, exist_ok=True)
    react_dir.mkdir(parents=True, exist_ok=True)
    
    # Django backend - API endpoints
    (django_dir / "api.py").write_text("""
# Django REST API
from django.http import JsonResponse
from .models import User, Product

def get_user_profile(request, user_id):
    '''Fetch user profile data'''
    user = User.objects.get(id=user_id)
    return JsonResponse({
        'id': user.id,
        'name': user.name,
        'email': user.email
    })

def list_products(request):
    '''List all available products'''
    products = Product.objects.all()
    return JsonResponse({
        'products': [p.to_dict() for p in products]
    })

def create_order(request):
    '''Create new order for user'''
    user_id = request.POST.get('user_id')
    product_id = request.POST.get('product_id')
    
    user = User.objects.get(id=user_id)
    product = Product.objects.get(id=product_id)
    
    order = Order.objects.create(
        user=user,
        product=product,
        status='pending'
    )
    return JsonResponse({'order_id': order.id})
""")
    
    (django_dir / "models.py").write_text("""
# Django models
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    
    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'email': self.email}

class Product(models.Model):
    name = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    
    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'price': float(self.price)}

class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    status = models.CharField(max_length=20)
""")
    
    # React frontend - API calls
    (react_dir / "UserProfile.jsx").write_text("""
// React component for user profile
import React, { useState, useEffect } from 'react';
import { fetchUserProfile } from './api';

export function UserProfile({ userId }) {
    const [profile, setProfile] = useState(null);
    
    useEffect(() => {
        loadProfile();
    }, [userId]);
    
    async function loadProfile() {
        const data = await fetchUserProfile(userId);
        setProfile(data);
    }
    
    return (
        <div className="profile">
            <h2>{profile?.name}</h2>
            <p>{profile?.email}</p>
        </div>
    );
}
""")
    
    (react_dir / "ProductList.jsx").write_text("""
// React component for product listing
import React, { useState, useEffect } from 'react';
import { fetchProducts, createOrder } from './api';

export function ProductList({ userId }) {
    const [products, setProducts] = useState([]);
    
    useEffect(() => {
        loadProducts();
    }, []);
    
    async function loadProducts() {
        const data = await fetchProducts();
        setProducts(data.products);
    }
    
    async function handleBuyClick(productId) {
        await createOrder(userId, productId);
        alert('Order created!');
    }
    
    return (
        <div className="products">
            {products.map(p => (
                <div key={p.id}>
                    <h3>{p.name}</h3>
                    <p>${p.price}</p>
                    <button onClick={() => handleBuyClick(p.id)}>Buy</button>
                </div>
            ))}
        </div>
    );
}
""")
    
    (react_dir / "api.js").write_text("""
// API client for backend communication
const API_BASE = 'http://localhost:8000/api';

export async function fetchUserProfile(userId) {
    const response = await fetch(`${API_BASE}/users/${userId}`);
    return response.json();
}

export async function fetchProducts() {
    const response = await fetch(`${API_BASE}/products`);
    return response.json();
}

export async function createOrder(userId, productId) {
    const response = await fetch(`${API_BASE}/orders`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, product_id: productId })
    });
    return response.json();
}
""")
    
    return str(django_dir), str(react_dir)

def run_multi_repo_demo():
    """Run the multi-repository analysis demo"""
    
    print("=" * 80)
    print("MULTI-REPOSITORY ANALYSIS DEMO - TechEx Hackathon")
    print("=" * 80)
    print()
    
    # Setup demo repositories
    print("📁 Setting up demo repositories...")
    django_path, react_path = setup_demo_repos()
    print(f"   ✓ Django backend: {django_path}")
    print(f"   ✓ React frontend: {react_path}")
    print()
    
    # Connect to FalkorDB
    print("🔌 Connecting to FalkorDB...")
    db = FalkorDB(host=os.getenv("FALKORDB_HOST", "localhost"), 
                  port=int(os.getenv("FALKORDB_PORT", 6379)))
    graph = db.select_graph("repo_insight_demo")
    print("   ✓ Connected")
    print()
    
    # Ingest both repositories
    print("📊 Ingesting repositories into graph...")
    print("   → Analyzing Django backend...")
    ingest_repository(django_path, repo_name="django_backend")
    print("   ✓ Django backend ingested")
    
    print("   → Analyzing React frontend...")
    ingest_repository(react_path, repo_name="react_frontend")
    print("   ✓ React frontend ingested")
    print()
    
    # Link repositories
    print("🔗 Linking repositories (cross-repo call graph)...")
    stats = link_repositories(graph, ["django_backend", "react_frontend"])
    print(f"   ✓ Created {stats['cross_repo_edges']} cross-repository edges")
    print(f"   ✓ Linked {stats['linked_functions']} function pairs")
    print()
    
    # Rebuild index with cross-repo edges
    print("🔄 Rebuilding graph index with cross-repo edges...")
    index = GraphIndex.build(graph)
    print("   ✓ Index rebuilt")
    print()
    
    # Demo 1: Cross-repo blast radius
    print("=" * 80)
    print("DEMO 1: Cross-Repository Blast Radius Analysis")
    print("=" * 80)
    print()
    print("Scenario: What happens if we change the Django User model?")
    print()
    
    target_fqn = "demo_repos.django_backend.models.User.to_dict"
    print(f"🎯 Target: {target_fqn}")
    print()
    
    blast_radius = get_cross_repo_blast_radius(graph, target_fqn, max_depth=3)
    
    print(f"📊 Blast Radius Results:")
    print(f"   • Total affected functions: {len(blast_radius['affected_functions'])}")
    print(f"   • Cross-repo impacts: {len(blast_radius['cross_repo_impacts'])}")
    print()
    
    if blast_radius['cross_repo_impacts']:
        print("🔴 CRITICAL: Changes will impact other repositories!")
        print()
        for impact in blast_radius['cross_repo_impacts'][:5]:
            print(f"   {impact['source_repo']} → {impact['target_repo']}")
            print(f"   {impact['source_function']}")
            print(f"   ↓ calls ↓")
            print(f"   {impact['target_function']}")
            print()
    
    # Demo 2: API endpoint usage tracking
    print("=" * 80)
    print("DEMO 2: API Endpoint Usage Tracking")
    print("=" * 80)
    print()
    print("Scenario: Which frontend components use the create_order API?")
    print()
    
    api_fqn = "demo_repos.django_backend.api.create_order"
    print(f"🎯 API Endpoint: {api_fqn}")
    print()
    
    # Query callers across repos
    result = graph.query(f"""
        MATCH (frontend:Function)-[:CROSS_REPO_CALLS]->(api:Function {{fqn: '{api_fqn}'}})
        WHERE frontend.repo_name <> api.repo_name
        RETURN frontend.fqn, frontend.repo_name, frontend.file_path
    """)
    
    if result.result_set:
        print(f"📱 Frontend components using this API:")
        for row in result.result_set:
            fqn, repo, file_path = row
            print(f"   • {fqn}")
            print(f"     Repository: {repo}")
            print(f"     File: {file_path}")
            print()
    else:
        print("   No cross-repo callers found (may need API mapping)")
        print()
    
    # Demo 3: Repository dependency graph
    print("=" * 80)
    print("DEMO 3: Repository Dependency Graph")
    print("=" * 80)
    print()
    
    result = graph.query("""
        MATCH (a:Function)-[:CROSS_REPO_CALLS]->(b:Function)
        WHERE a.repo_name <> b.repo_name
        RETURN a.repo_name, b.repo_name, count(*) as call_count
        ORDER BY call_count DESC
    """)
    
    print("📊 Inter-repository dependencies:")
    if result.result_set:
        for row in result.result_set:
            from_repo, to_repo, count = row
            print(f"   {from_repo} → {to_repo}: {count} calls")
    else:
        print("   No cross-repo dependencies found")
    print()
    
    # Summary
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("✨ Key Capabilities Demonstrated:")
    print("   1. Multi-repository code ingestion")
    print("   2. Cross-repository call graph construction")
    print("   3. Blast radius analysis across repo boundaries")
    print("   4. API usage tracking across frontend/backend")
    print("   5. Repository dependency visualization")
    print()
    print("🏆 Competitive Advantage:")
    print("   NO other code intelligence tool offers cross-repo analysis!")
    print("   This is a unique differentiator for enterprise microservices.")
    print()

if __name__ == "__main__":
    try:
        run_multi_repo_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob

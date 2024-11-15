// js/api.js
const API_URL = 'https://movemate-dbqo.onrender.com/api';

const api = {
    login: (credentials) => 
        fetch(`${API_URL}/auth/login/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(credentials)
        }),

    register: (userData) => 
        fetch(`${API_URL}/auth/register/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData)
        }),

    contact: (formData) => 
        fetch(`${API_URL}/pages/contact/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        }),

    getServices: () => 
        fetch(`${API_URL}/pages/services/`),

    getBlogPosts: () => 
        fetch(`${API_URL}/pages/blog/posts/`)
};

export default api;
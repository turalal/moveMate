# MoveMate API

Backend REST API for MoveMate website built with Django REST Framework.

## Requirements

- Python 3.8+
- PostgreSQL
- pip

## Installation

1. Clone repository:
```bash
git clone https://github.com/Ismat-Samadov/moveMate.git
cd movemate
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install django
pip install djangorestframework
pip install django-cors-headers
pip install djangorestframework-simplejwt
pip install psycopg2-binary
pip install python-dotenv
pip install Pillow
pip install django-filter
```

4. Create `.env` file:
```
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

DB_NAME=movemate
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
```

5. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

6. Create superuser:
```bash
python manage.py createsuperuser
```

7. Run server:
```bash
python manage.py runserver
```

## API Endpoints

### Authentication

#### Register
```bash
curl -X POST http://localhost:8000/api/auth/register/ \
-H "Content-Type: application/json" \
-d '{"username":"testuser","password":"testpass123","email":"test@test.com"}'
```

#### Login
```bash
curl -X POST http://localhost:8000/api/auth/login/ \
-H "Content-Type: application/json" \
-d '{"username":"testuser","password":"testpass123"}'
```

#### Refresh Token
```bash
curl -X POST http://localhost:8000/api/auth/token/refresh/ \
-H "Content-Type: application/json" \
-d '{"refresh":"your_refresh_token"}'
```

### Services

#### List Services
```bash
curl http://localhost:8000/api/pages/services/
```

#### Get Service Detail
```bash
curl http://localhost:8000/api/pages/services/1/
```

### Blog

#### List Categories
```bash
curl http://localhost:8000/api/pages/blog/categories/
```

#### List Posts
```bash
curl http://localhost:8000/api/pages/blog/posts/
```

#### Get Post Detail
```bash
curl http://localhost:8000/api/pages/blog/posts/post-slug/
```

#### Filter Posts
```bash
# By category
curl http://localhost:8000/api/pages/blog/posts/?category=category-slug

# By search term
curl http://localhost:8000/api/pages/blog/posts/?search=keyword
```

### Comments

#### Create Comment (Requires Authentication)
```bash
curl -X POST http://localhost:8000/api/pages/blog/posts/post-slug/comments/ \
-H "Authorization: Bearer your_access_token" \
-H "Content-Type: application/json" \
-d '{"content":"Test comment"}'
```

#### Delete Comment (Requires Authentication)
```bash
curl -X DELETE http://localhost:8000/api/pages/blog/comments/1/ \
-H "Authorization: Bearer your_access_token"
```

### Contact

#### Send Message
```bash
curl -X POST http://localhost:8000/api/pages/contact/ \
-H "Content-Type: application/json" \
-d '{"name":"Test User","email":"test@test.com","subject":"Test Subject","message":"Test message"}'
```

## Features

- JWT Authentication
- Blog with categories and comments
- Services management
- Contact form
- Image upload support
- Filtering and search functionality
- Pagination
- Test coverage

## Development

Run tests:
```bash
python manage.py test
```

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request
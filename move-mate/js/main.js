(function ($) {
    "use strict";

    const API_URL = 'https://movemate-dbqo.onrender.com/api';
    let authToken = localStorage.getItem('token');

    // Spinner
    var spinner = function () {
        setTimeout(function () {
            if ($('#spinner').length > 0) {
                $('#spinner').removeClass('show');
            }
        }, 1);
    };
    spinner(0);
    
    // Initiate the wowjs
    new WOW().init();

    // Sticky Navbar
    $(window).scroll(function () {
        if ($(this).scrollTop() > 45) {
            $('.nav-bar').addClass('sticky-top shadow-sm');
        } else {
            $('.nav-bar').removeClass('sticky-top shadow-sm');
        }
    });

    // Facts counter
    $('[data-toggle="counter-up"]').counterUp({
        delay: 5,
        time: 2000
    });

    // Modal Video
    $(document).ready(function () {
        var $videoSrc;
        $('.btn-play').click(function () {
            $videoSrc = $(this).data("src");
        });
        console.log($videoSrc);

        $('#videoModal').on('shown.bs.modal', function (e) {
            $("#video").attr('src', $videoSrc + "?autoplay=1&amp;modestbranding=1&amp;showinfo=0");
        })

        $('#videoModal').on('hide.bs.modal', function (e) {
            $("#video").attr('src', $videoSrc);
        })
    });

    // Testimonial-carousel
    $(".testimonial-carousel").owlCarousel({
        autoplay: true,
        smartSpeed: 2000,
        center: false,
        dots: false,
        loop: true,
        margin: 25,
        nav : true,
        navText : [
            '<i class="bi bi-arrow-left"></i>',
            '<i class="bi bi-arrow-right"></i>'
        ],
        responsiveClass: true,
        responsive: {
            0:{
                items:1
            },
            576:{
                items:1
            },
            768:{
                items:2
            },
            992:{
                items:2
            },
            1200:{
                items:2
            }
        }
    });
    
    // Back to top button
    $(window).scroll(function () {
        if ($(this).scrollTop() > 300) {
            $('.back-to-top').fadeIn('slow');
        } else {
            $('.back-to-top').fadeOut('slow');
        }
    });
    $('.back-to-top').click(function () {
        $('html, body').animate({scrollTop: 0}, 1500, 'easeInOutExpo');
        return false;
    });

    // API Handlers
    function handleResponse(response) {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }

    // Contact Form Handler
    $('#contactForm').submit(function(e) {
        e.preventDefault();
        const formData = {
            name: $('#name').val(),
            email: $('#email').val(),
            subject: $('#subject').val() || 'Contact Form Submission',
            message: $('#message').val()
        };

        fetch(`${API_URL}/pages/contact/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        })
        .then(handleResponse)
        .then(data => {
            if (data.message === "Message sent successfully") {
                window.location.href = 'thank_you.html';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to send message. Please try again.');
        });
    });

    // Login Form Handler
    $('#loginForm').submit(function(e) {
        e.preventDefault();
        const credentials = {
            username: $('#username').val(),
            password: $('#password').val()
        };

        fetch(`${API_URL}/auth/login/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(credentials)
        })
        .then(handleResponse)
        .then(data => {
            if (data.access) {
                localStorage.setItem('token', data.access);
                authToken = data.access;
                window.location.href = 'dashboard.html';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Login failed. Please check your credentials.');
        });
    });

    // Register Form Handler
    $('#registerForm').submit(function(e) {
        e.preventDefault();
        const userData = {
            username: $('#registerUsername').val(),
            email: $('#registerEmail').val(),
            password: $('#registerPassword').val(),
            phone: $('#registerPhone').val()
        };

        fetch(`${API_URL}/auth/register/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData)
        })
        .then(handleResponse)
        .then(data => {
            alert('Registration successful! Please login.');
            window.location.href = 'login.html';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Registration failed. Please try again.');
        });
    });

    // Load Services
    function loadServices() {
        if ($('#services-container').length) {
            fetch(`${API_URL}/pages/services/`)
                .then(handleResponse)
                .then(data => {
                    const services = data.results;
                    const container = $('#services-container');
                    container.empty();
                    services.forEach(service => {
                        container.append(`
                            <div class="col-lg-4 col-md-6 wow fadeInUp" data-wow-delay="0.1s">
                                <div class="service-item p-4">
                                    ${service.icon ? `<img src="${service.icon}" class="img-fluid mb-4" alt="${service.title}">` : ''}
                                    <h5 class="mb-3">${service.title}</h5>
                                    <p>${service.description}</p>
                                </div>
                            </div>
                        `);
                    });
                })
                .catch(error => console.error('Error loading services:', error));
        }
    }

    // Load Blog Posts
    function loadBlogPosts() {
        if ($('#blog-container').length) {
            fetch(`${API_URL}/pages/blog/posts/`)
                .then(handleResponse)
                .then(data => {
                    const posts = data.results;
                    const container = $('#blog-container');
                    container.empty();
                    posts.forEach(post => {
                        container.append(`
                            <div class="col-lg-4 col-md-6 wow fadeInUp" data-wow-delay="0.1s">
                                <div class="blog-item bg-light rounded overflow-hidden">
                                    <div class="blog-img position-relative overflow-hidden">
                                        <img class="img-fluid" src="${post.image || 'img/blog-1.jpg'}" alt="">
                                    </div>
                                    <div class="p-4">
                                        <div class="d-flex mb-3">
                                            <small class="me-3"><i class="far fa-user text-primary me-2"></i>${post.author_name}</small>
                                            <small><i class="far fa-calendar text-primary me-2"></i>${new Date(post.created_at).toLocaleDateString()}</small>
                                        </div>
                                        <h5 class="mb-3">${post.title}</h5>
                                        <p>${post.content.substring(0, 100)}...</p>
                                        <a class="text-uppercase" href="blog-detail.html?slug=${post.slug}">Read More <i class="bi bi-arrow-right"></i></a>
                                    </div>
                                </div>
                            </div>
                        `);
                    });
                })
                .catch(error => console.error('Error loading blog posts:', error));
        }
    }

    // Load Blog Post Detail
    function loadBlogPostDetail() {
        const urlParams = new URLSearchParams(window.location.search);
        const slug = urlParams.get('slug');
        
        if (slug && $('#blog-detail-container').length) {
            fetch(`${API_URL}/pages/blog/posts/${slug}/`)
                .then(handleResponse)
                .then(post => {
                    $('#blog-detail-container').html(`
                        <div class="mb-5">
                            <img class="img-fluid w-100 rounded mb-5" src="${post.image || 'img/blog-1.jpg'}" alt="">
                            <h1 class="mb-4">${post.title}</h1>
                            <div class="d-flex mb-3">
                                <small class="me-3"><i class="far fa-user text-primary me-2"></i>${post.author_name}</small>
                                <small><i class="far fa-calendar text-primary me-2"></i>${new Date(post.created_at).toLocaleDateString()}</small>
                            </div>
                            <p>${post.content}</p>
                        </div>
                    `);
                })
                .catch(error => console.error('Error loading blog post:', error));
        }
    }

    // Comment Form Handler
    $('#commentForm').submit(function(e) {
        e.preventDefault();
        if (!authToken) {
            alert('Please login to comment');
            return window.location.href = 'login.html';
        }

        const urlParams = new URLSearchParams(window.location.search);
        const slug = urlParams.get('slug');
        const content = $('#comment').val();

        fetch(`${API_URL}/pages/blog/posts/${slug}/comments/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
            },
            body: JSON.stringify({ content })
        })
        .then(handleResponse)
        .then(data => {
            alert('Comment posted successfully!');
            location.reload();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to post comment. Please try again.');
        });
    });

    // Initialize API-dependent content
    $(document).ready(function() {
        loadServices();
        loadBlogPosts();
        loadBlogPostDetail();
    });

    $('.owl-carousel').owlCarousel({
    items: 1, // Hər dəfə bir testimonial göstərin
    loop: true,
    autoplay: true,
    autoplayTimeout: 5000,
    smartSpeed: 800,
});


})(jQuery);

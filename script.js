document.addEventListener('DOMContentLoaded', () => {
    const signupForm = document.getElementById('signup-form');
    const loginForm = document.getElementById('login-form');
    const sellLink = document.getElementById('sell-link');
    const signupLink = document.getElementById('signup-link');
    const loginLink = document.getElementById('login-link');
    const logoutLink = document.getElementById('logout-link');
    const buyLinks = document.querySelectorAll('.buy-link');

    // Check if user is logged in
    auth.onAuthStateChanged(user => {
        if (user) {
            signupLink.style.display = 'none';
            loginLink.style.display = 'none';
            logoutLink.style.display = 'block';
        } else {
            signupLink.style.display = 'block';
            loginLink.style.display = 'block';
            logoutLink.style.display = 'none';
        }
    });

    // Handle signup form submission
    if (signupForm) {
        signupForm.addEventListener('submit', function(event) {
            event.preventDefault();

            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;

            auth.createUserWithEmailAndPassword(email, password)
                .then(userCredential => {
                    alert('Signup successful! Please log in.');
                    window.location.href = 'login.html';
                })
                .catch(error => {
                    alert(error.message);
                });
        });
    }

    // Handle login form submission
    if (loginForm) {
        loginForm.addEventListener('submit', function(event) {
            event.preventDefault();

            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;

            auth.signInWithEmailAndPassword(email, password)
                .then(userCredential => {
                    alert('Login successful!');
                    window.location.href = 'index.html';
                })
                .catch(error => {
                    alert(error.message);
                });
        });
    }

    // Handle logout
    if (logoutLink) {
        logoutLink.addEventListener('click', function(event) {
            event.preventDefault();
            auth.signOut().then(() => {
                alert('Logged out successfully!');
                window.location.href = 'index.html';
            });
        });
    }

    // Restrict sell and buy links to logged-in users
    if (sellLink) {
        sellLink.addEventListener('click', function(event) {
            if (!auth.currentUser) {
                event.preventDefault();
                alert('You must be logged in to sell a product.');
                window.location.href = 'login.html';
            }
        });
    }

    if (buyLinks) {
        buyLinks.forEach(link => {
            link.addEventListener('click', function(event) {
                if (!auth.currentUser) {
                    event.preventDefault();
                    alert('You must be logged in to buy a product.');
                    window.location.href = 'login.html';
                }
            });
        });
    }
});

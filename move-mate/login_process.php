<?php
// login_process.php
session_start();

// Database configuration
$servername = "localhost";
$username = "root"; // your DB username
$password = ""; // your DB password
$dbname = "movemate"; // your DB name

// Create a new database connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check for connection errors
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Get the email and password from the form
$email = $conn->real_escape_string($_POST['email']);
$password = $_POST['password'];

// Check if user exists
$sql = "SELECT * FROM users WHERE email = '$email'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    // Fetch user data
    $user = $result->fetch_assoc();

    // Verify password
    if (password_verify($password, $user['password'])) {
        // Store user info in session variables
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['user_name'] = $user['name'];
        $_SESSION['user_email'] = $user['email'];

        // Redirect to dashboard or homepage
        header("Location: dashboard.html");
        exit();
    } else {
        echo "Incorrect password. <a href='login.html'>Try again</a>";
    }
} else {
    echo "User not found. <a href='signup.html'>Sign up</a>";
}

// Close the database connection
$conn->close();
?>

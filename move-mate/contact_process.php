<?php
// contact_process.php
session_start();

// Output buffering aktivləşdirin
ob_start();

// Debugging aktivləşdirin
error_reporting(E_ALL);
ini_set('display_errors', 1);

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

// Get the form data and sanitize
$name = $conn->real_escape_string(trim($_POST['name']));
$email = $conn->real_escape_string(trim($_POST['email']));
$message = $conn->real_escape_string(trim($_POST['message']));

// Email sending operation
$to = "sales@movemate.me"; // Replace with your email address
$subject = "New Message from $name";
$body = "Name: $name\nEmail: $email\n\nMessage:\n$message";
$headers = "From: $email";

// Mail funksiyasının işini yoxlayın
if (mail($to, $subject, $body, $headers)) {
    // E-poçt göndərilibsə yönləndir
    header("Location: thank_you.html");
    ob_end_flush(); // Output buffering bitir
    exit();
} else {
    // Mail funksiyası uğursuz olarsa, debugging üçün mesaj verin
    echo "There was an issue sending the email. Please check your server's email configuration.";
}

// Close the database connection
$conn->close();
ob_end_flush(); // Output buffering bitir
?>

<?php
// signup_process.php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
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

    // Prepare form inputs for database insertion
    $name = $conn->real_escape_string($_POST['name']);
    $email = $conn->real_escape_string($_POST['email']);
    $password = password_hash($_POST['password'], PASSWORD_BCRYPT); // Hashing for security
    $phone = $conn->real_escape_string($_POST['phone']);
    $membership = $conn->real_escape_string($_POST['membership']);

    // Insert form data into the database
    $sql = "INSERT INTO users (name, email, password, phone, membership) VALUES ('$name', '$email', '$password', '$phone', '$membership')";

    if ($conn->query($sql) === TRUE) {
        echo "Sign up successful!";
        header("Location: welcome.html"); // Redirect to a welcome or success page
        exit();
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }

    // Close the database connection
    $conn->close();
}
?>

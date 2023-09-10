
# 1. Github Pages Tutorial

https://pages.github.com/

# 2. A Simple Example

## Directory Structure

Your GitHub repository might look something like this:

    /my-github-pages-repo
    ├── index.html
    ├── style.css
    └── script.js

## HTML (index.html)

    This is your main HTML file. Save this content in a file called index.html.
    
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>My GitHub Pages</title>
        <link rel="stylesheet" href="style.css">
    </head>
    <body>
        <header>
            <h1>Welcome to My GitHub Pages</h1>
        </header>
        
        <main>
            <p>This is a sample paragraph.</p>
            <button id="sample-button">Click Me</button>
        </main>
    
        <footer>
            <p>Created with ❤️ by [Your Name]</p>
        </footer>
    
        <script src="script.js"></script>
    </body>
    </html>

## CSS (style.css)

Here's a basic stylesheet. Save this content in a file called style.css.

    /* Reset some default browser styles */
    body, h1, h2, h3, p, button {
        margin: 0;
        padding: 0;
    }
    
    /* Add a background color to the page */
    body {
        background-color: #f4f4f4;
        font-family: Arial, sans-serif;
    }
    
    /* Style header */
    header {
        background-color: #50b3a2;
        color: white;
        text-align: center;
        padding: 1em;
    }
    
    /* Style main content */
    main {
        padding: 1em;
    }
    
    /* Style footer */
    footer {
        background-color: #333;
        color: white;
        text-align: center;
        padding: 1em;
    }

## JavaScript (script.js)

Here's some basic JavaScript to make your page interactive. Save this content in a file called script.js.

    document.addEventListener("DOMContentLoaded", function() {
        const button = document.getElementById("sample-button");
        button.addEventListener("click", function() {
            alert("Hello, World!");
        });
    });

## Push to GitHub and Enable GitHub Pages

1. Create a new GitHub repository or use an existing one.
2. Add these files to your repository.
3. Commit and push the changes to GitHub.
4. Go to the repository settings on GitHub.
4. Scroll down to the GitHub Pages section.
5. Choose the branch you want to use (usually main or gh-pages).
6. Your site should be accessible at https://<username>.github.io/<repository>/.







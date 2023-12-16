
# 1. Github Pages Tutorial

https://pages.github.com/

# 2. A Simple Example

## Directory Structure

Your GitHub repository might look something like this:

    /my-github-pages-repo
    ├── index.html
    ├── style.css

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
            <!-- About Me section starts here -->
            <section id="about-me">
                <h2>About Me</h2>
                <p>Hi, my name is Shiteng Yang. I'm currently working as a data scientist at Rocket Companies.</p>
                <p>Welcome to my GitHub Pages, and feel free to review my projects.</p>
                <p>My Linkedin: <a href="https://www.linkedin.com/in/shiteng-yang-173939102/" target="_blank">shiteng-yang-linkedin</a></p>
                <p>My Email: <a href="mailto:yangshiteng007@gmail.com">yangshiteng007@gmail.com</a></p>
            </section>
            <!-- About Me section ends here -->
    
        </main>
    
        <footer>
            <p>Created with ❤️ by Shiteng Yang</p>
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
    
    /* Style for About Me section */
    #about-me {
        margin-top: 30px;
        padding: 15px;
        background-color: #e9e9e9;
        border-radius: 5px;
    }
    
    #about-me h2 {
        margin-bottom: 15px;  /* Increase this to add more space */
    }


## Push to GitHub and Enable GitHub Pages

1. Create a new GitHub repository or use an existing one.
2. Add these files to your repository.
3. Commit and push the changes to GitHub.
4. Go to the repository settings on GitHub.
4. Scroll down to the GitHub Pages section.
5. Choose the branch you want to use (usually main or gh-pages).
6. Your site should be accessible at https://<username>.github.io/<repository>/.

## Test it in local environment

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1b71b0df-2bdd-4826-92d6-35560fc44e1c)






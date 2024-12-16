# Deployment Guide

## Overview
This document outlines the steps and key components required to deploy the application on the web.

---

## File Structure
The deployment consists of the following two primary files:

1. **Static Files**
   - Contains CSS, images, and HTML files used for the front-end design and layout.

2. **JavaScript Files**
   - Contains logic and interactions for the front-end, enabling dynamic behavior on the website.

Ensure both these files are correctly placed in the appropriate directories for the web server to serve them.

---

## Database

The application uses **Neo4j** as its database. Below are the details for setting up and configuring the database:

1. **Installation**
   - Install Neo4j from the official website: [https://neo4j.com/download/](https://neo4j.com/download/)
   - Follow the installation instructions for your operating system.

2. **Configuration**
   - Set up your database with the appropriate credentials and connect it to the application.
   - Update the database connection settings in the application configuration file, typically `config.py` or `.env`.

3. **Data Import**
   - Load your graph data into the Neo4j instance before starting the application.

4. **Access Sandbox**
   - Use the following link to access the Neo4j sandbox: [https://sandbox.neo4j.com/?usecase=bloom](https://sandbox.neo4j.com/?usecase=bloom)
   - Enter your login credentials to view and manage the data.

---

## Running the Application

To run the application, execute the following command in your terminal:

```bash
python app.py
```

### Prerequisites
- Ensure Python is installed on your machine.
- Install all required Python dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```
- Confirm that the Neo4j database is running and accessible.

---

## Deployment Notes
- Ensure that the web server is configured to serve the static and JavaScript files correctly.
- Verify that the Neo4j database is properly connected and populated with necessary data.
- Use a virtual environment to manage Python dependencies to avoid version conflicts.

---

## Support
For any issues during deployment, refer to the application documentation or contact the development team.



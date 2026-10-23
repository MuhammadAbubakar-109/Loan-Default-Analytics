## Loan Default Risk Check — Frontend

A web-based frontend for a loan default risk assessment application, built with React, TypeScript, and Vite. The interface provides dedicated pages for loan risk prediction and analysis, communicating with a backend API to support the application's workflow.

### Overview

The frontend serves as the user interface for the Loan Default Risk Check application. It is responsible for rendering application pages, managing user interactions, and integrating with the backend API.

### Key Features
Loan Risk Prediction: User interface for the loan default prediction workflow.

Risk Analysis: Dedicated analysis page for presenting loan-related analytical information.

API Integration: Centralized API module for communication with the backend service.

Type Safety: TypeScript support for more maintainable and reliable frontend code.

Modern Development Workflow: Vite-powered development server and production build tooling.

### Technology Stack

React — Component-based user interface development.
TypeScript — Static typing and improved code maintainability.
Vite — Frontend development server and production build tool.
CSS — Application styling and layout.
npm — Package management and script execution.

### Project Structure

Frontend/
├── src/
│   ├── AnalysisPage.tsx  # Analysis page
│   ├── App.tsx           # Main application component
│   ├── PredictPage.tsx   # Loan prediction page
│   ├── api.ts            # Backend API integration
│   ├── fields.ts         # Shared field definitions
│   ├── data.txt          # Application data or supporting content
│   ├── main.tsx          # Frontend entry point
│   └── styles.css        # Global/application styles
├── .env.example          # Example environment configuration
├── index.html            # HTML entry point
├── package.json          # Dependencies and npm scripts
├── package-lock.json     # Locked dependency versions
├── tsconfig.json         # TypeScript configuration
├── vite.config.ts        # Vite configuration
└── README.md             # Frontend documentation

### Configure Environment Variables

Review .env.example for the environment variables supported by the project.

If the frontend requires a backend API URL, configure the appropriate variable in a local .env file according to the existing application configuration.

Keep local secrets and environment-specific settings out of version control. Only expose values intended for use in the browser.

### Application Architecture

The frontend is organized into focused modules:

App.tsx — Main application component and page composition.

PredictPage.tsx — User interface for the prediction workflow.

AnalysisPage.tsx — User interface for loan risk analysis.

api.ts — API communication logic, keeping network requests separate from page components.

fields.ts — Shared field definitions used by the application.

styles.css — Styling for the frontend interface.

main.tsx — Entry point responsible for initializing the React application.


### Backend Integration

The frontend communicates with the backend through the API integration module in src/api.ts.

For successful integration:

Start the backend service.
Verify the configured API base URL.
Confirm that the frontend request formats match the backend API contract.

Ensure the backend is reachable and any required cross-origin (CORS) configuration is in place.


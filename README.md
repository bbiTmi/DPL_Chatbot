cd .\frontend\

npm install

npm run dev

| File               | Description                                                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MainPage.jsx`     | The main landing page where users choose their role: **Customer** or **Manager**. Routes accordingly.                                                        |
| `MainPage.css`     | Contains styles specifically for the `MainPage.jsx` layout, alignment, and button appearance.                                                                |
| `FaceAuth.jsx`     | The webcam-based face authentication page for customers. Captures and sends face image to backend.                                                           |
| `ManagerLogin.jsx` | A login screen placeholder for managers. Typically includes form-based authentication.                                                                       |
| `Webcam.js`        | A custom JavaScript class that wraps the browser's webcam API. Provides methods for starting/stopping the stream, flipping the camera, and taking snapshots. |

| File        | Description                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------- |
| `App.jsx`   | The root React component where routing and top-level layout logic is managed.                                   |
| `App.css`   | Global or shared styles for the app. Applies layout, fonts, and default visuals.                                |
| `index.css` | Global baseline styles and resets for HTML, body, and other tags. Ensures consistent rendering across browsers. |
| `main.jsx`  | The app's entry point. Mounts the `<App />` component to the DOM and initializes React.                         |

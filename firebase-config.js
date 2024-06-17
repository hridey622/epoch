// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyC3Lx60_1N4MLqUr9zsWx8KeVPxg4e54lg",
    authDomain: "epoch-b01f2.firebaseapp.com",
    projectId: "epoch-b01f2",
    storageBucket: "epoch-b01f2.appspot.com",
    messagingSenderId: "138705889507",
    appId: "YOUR_APP_ID"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
const db = firebase.firestore();

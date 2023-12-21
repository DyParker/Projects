import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
    apiKey: "AIzaSyDPrvNRiqlpUh4Tlqh3lrBTZOqwmvTg6QI",
    authDomain: "mealswipes-f0902.firebaseapp.com",
    projectId: "mealswipes-f0902",
    storageBucket: "mealswipes-f0902.appspot.com",
    messagingSenderId: "1016636753973",
    appId: "1:1016636753973:web:cf9bc5028c130674a6097c",
    measurementId: "G-E1M9GX8M3E"
};





//Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

//Initialize Firebase Auth
export const auth = getAuth(app);

//Initialize db

export const db = getFirestore(app);
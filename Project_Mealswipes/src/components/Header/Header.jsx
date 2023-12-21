import './Header.css';

import { Link } from "react-router-dom";

import { doc, getDoc } from "firebase/firestore";
import { db } from "../../firebase";

import React, { useEffect, useState } from 'react';
import { auth } from '../../firebase';



export default function Header() {
    const [curr_uid, setCurr_UID] = useState('')
    const [curr_email, setCurr_Email] = useState('')

    useEffect(() => {
        const unsub = auth.onAuthStateChanged((authObj) => {
            unsub();
            if (authObj) {
                setCurr_UID(authObj.uid);
                setCurr_Email(authObj.email)
            } else {
                // not logged in
            }
        });
    }, []);

    const user = curr_uid;
    const [authUser, setAuthUser] = useState(null);
    const [role, setRole] = useState('');
    const [phoneNum, setPhoneNum] = useState('');
    const [name, setName] = useState('');

    async function userInfo(user) {
        const docRef = doc(db, "users", user);
        const docSnap = await getDoc(docRef);
        setRole(String(docSnap.data().role));
        setPhoneNum(String(docSnap.data().phoneNum));
        setName(String(docSnap.data().name));
    }

    useEffect(() => {
        if (user) {
            userInfo(user);
        }
        else {
            setRole('logged-out');
        }
    }, [user]);




    return (
        <>

            <nav className="navbar navbar-expand-lg fixed-top navbar-light bg-light">
                <div className="container-fluid">
                    <div className="navbar-brand"><Link to='/'>MealSwipes</Link></div>
                    <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span className="navbar-toggler-icon" />
                    </button>
                    <div className="collapse navbar-collapse" id="navbarSupportedContent">
                        <ul className="navbar-nav me-auto mb-2 mb-lg-0">
                            <li className="nav-item">
                                <Link to='/'>Home</Link>
                            </li>
                            <li className="nav-item">
                                <Link to='/about'>About</Link>
                            </li>
                            <li className="nav-item">
                                <Link to='/signin'>Sign In</Link>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>




        </>
    )
}
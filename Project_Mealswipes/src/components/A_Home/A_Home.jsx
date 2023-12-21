import V_Header from '../Volunteer_Header/Volunteer_Header.jsx';
import D_Header from '../Donor_Header/Donor_Header.jsx';
import S_Header from '../Shelter_Header/Shelter_Header.jsx';

import { doc, getDoc } from "firebase/firestore";
import { db } from "../../firebase";

import React, { useEffect, useState } from 'react';
import { auth } from '../../firebase';


export default function Home() {
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
    }, [user]);
    let headerComponent;
    switch (role) {
        case 'volunteer':
            headerComponent = <V_Header />;
            break;
        case 'donor':
            headerComponent = <D_Header />;
            break;
        case 'shelter':
            headerComponent = <S_Header />;
            break;
    }


    return (
        <>
            {headerComponent}


            <section id="section0" className="container-fluid">
                <div className="container">
                    <div className="row">
                        <div className="col align-self-center">
                            <h2>WELCOME TO PROJECT MEALSWIPES </h2>
                            <p>We are the solution to the 119 billion pounds of annual food waste in the US.</p>
                        </div>
                    </div>
                </div>
            </section>
        </>
    )
}
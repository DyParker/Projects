import Donor_Header from '../Donor_Header/Donor_Header.jsx';
import './Donor_Info.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowCircleLeft } from '@fortawesome/free-solid-svg-icons';
import { doc, getDoc } from "firebase/firestore";
import { db } from "../../firebase";
import { Link } from "react-router-dom";
import c_img from './contact_image.png';
import ChangePassword from '../V_Profile_Info/ChangePassword.jsx';

import React, { useEffect, useState } from 'react';
import { auth } from '../../firebase';
//<FilePerson className="centerFile"/>

export default function Donor_Info() {
    const[curr_uid, setCurr_UID] = useState('')
    const[curr_email, setCurr_Email] = useState('')
    const[showModal, setShowModal] = useState(false)

    useEffect(() => {
        const unsub = auth.onAuthStateChanged((authObj) => {
            unsub();
            if (authObj) {
                setCurr_UID(authObj.uid);
                setCurr_Email(authObj.email);
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
    const [tasksCompleted, setTasksCompleted] = useState('');

    async function userInfo(user) {
    const docRef = doc(db, "users", user);
    const docSnap = await getDoc(docRef);
    setRole(String(docSnap.data().role));
    setPhoneNum(String(docSnap.data().phoneNum));
    setName(String(docSnap.data().name));
    setTasksCompleted(docSnap.data().tasksCompleted);
}

useEffect(() => {
    if (user) {
        userInfo(user);
    }
}, [user]);

    return (
        <body>
            <Donor_Header />
            <section className="vh-100 profile-info">
                <section className="push"></section>
                <div className="container py-5 h-100">
                    <div className="row backarrow"><Link to='/donorui'><FontAwesomeIcon icon={faArrowCircleLeft} color="grey" size="2xl" /></Link></div>
                    <div className="row d-flex justify-content-center align-items-center h-100 mt-0">
                        <div className="col col-lg-6 mb-4 mb-lg-0">
                            <div className="card mb-3 profile-card">
                                <div className="row g-0">
                                    <div className="col-md-4 gradient-custom text-center text-white profile-pic">
                                        <img src={c_img} className="img-fluid my-5" />
                                        <p>Name: {name}</p>
                                        <p>Role: {role}</p>
                                        <button type="button" className="btn btn-outline-dark" onClick={() => setShowModal(true)}>Change Password</button>
                                    </div>
                                    <div className="col-md-8">
                                        <div className="card-body p-4">
                                            <h6>Information</h6>
                                            <hr className="mt-0 mb-4" />
                                            <div className="row pt-1">
                                                <div className="col-6 mb-3 email">
                                                    <h6>Email</h6>
                                                    <p>{curr_email}</p>
                                                </div>
                                                <div className="col-6 mb-3">
                                                    <h6>Phone Number</h6>
                                                    <p>{phoneNum}</p>
                                                </div>
                                            </div>
                                            <h6>Past Trips</h6>
                                            <hr className="mt-0 mb-4" />
                                            <div className="row pt-1">
                                                <div className="col-6 mb-3">
                                                    <h6>Meals Delivered</h6>
                                                    <p>{tasksCompleted}</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
            <ChangePassword show={showModal} onClose={()=>setShowModal(false)}></ChangePassword>
        </body>
    )
}
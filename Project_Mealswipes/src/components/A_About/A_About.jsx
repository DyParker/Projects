import V_Header from '../Volunteer_Header/Volunteer_Header.jsx';
import D_Header from '../Donor_Header/Donor_Header.jsx';
import S_Header from '../Shelter_Header/Shelter_Header.jsx';

import chart from './Food_Waste_US_Chart.png';

import { doc, getDoc } from "firebase/firestore";
import { db } from "../../firebase";

import React, { useEffect, useState } from 'react';
import { auth } from '../../firebase';


export default function A_About() {

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
      
      <section id="aboutus">
        <div className="col section-header my-2 mx-0">FOOD WASTE IN THE U.S</div>
        <div className="row">
          <div className="col food-waste">
            <p>Food waste is an avoidable problem that results from inadequate resources put towards getting leftover food to people who need it most. We believe many people would be willing to dedicate their time and energy to helping their communities if we lower the information barrier, create social incentives, and raise awareness of food insecurity.</p>
          </div>
          <div className="col food-waste-pic">
            <img src={chart} alt={"piechart"} id="aboutuspic" />
          </div>
        </div>
        <div className="row mission-statement my-2">
          <p>Our mission is to reduce hunger and increase food sustainability in our communities.</p>
        </div>
      </section>
    </>
  )
}
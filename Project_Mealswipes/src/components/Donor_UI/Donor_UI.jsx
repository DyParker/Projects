import Donor_Header from '../Donor_Header/Donor_Header.jsx';
import './Donor_UI.css';
import Donor_Timeline from './Donor_Timeline.jsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlusCircle, faClipboardCheck, faComment } from '@fortawesome/free-solid-svg-icons'
// import ReactDOM from 'react-dom'
import React, { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { collection, getDoc, getDocs, query, where, doc, setDoc } from 'firebase/firestore';

import firebase from 'firebase/compat/app';
import 'firebase/compat/firestore';
import 'firebase/compat/auth';
import 'firebase/compat/analytics';
import { db } from "../../firebase";

import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';


firebase.initializeApp({
  // your config
  apiKey: "AIzaSyDPrvNRiqlpUh4Tlqh3lrBTZOqwmvTg6QI",
  authDomain: "mealswipes-f0902.firebaseapp.com",
  projectId: "mealswipes-f0902",
  storageBucket: "mealswipes-f0902.appspot.com",
  messagingSenderId: "1016636753973",
  appId: "1:1016636753973:web:cf9bc5028c130674a6097c",
  measurementId: "G-E1M9GX8M3E"
})
const auth = firebase.auth();

const firestore = firebase.firestore();




export default function Donor_UI() {

  
 
  

  const navigate = useNavigate();
  const tasksRef = firestore.collection('tasks');
  const [formValue, setFormValue] = useState({ title: '', quantity: '', description: '', availability: '12:00 PM' });
  const [loading, setLoading] = useState(false);

  const getUserAddress = async (uid) => {
    const userRef = firestore.collection('users').doc(uid);
    const userDoc = await userRef.get();
    
    if (!userDoc.exists) {
      throw new Error(`User with uid ${uid} not found`);
    }
    
    const { address } = userDoc.data();
    return address;
  };


  const sendTask = async (e) => {
    e.preventDefault();
    const newTaskRef = doc(tasksRef);
    const d_address = await getUserAddress(auth.currentUser.uid);
    
    await setDoc(newTaskRef, {
      title: formValue.title,
      quantity: formValue.quantity,
      description: formValue.description,
      availability: formValue.availability,
      createdAt: firebase.firestore.FieldValue.serverTimestamp(),
      d_uid: auth.currentUser.uid,
      status: 'available',
      ref: newTaskRef.id,
      d_address: d_address,
      s_address: ''
    });
   
    setFormValue({ title: '', quantity: '', description: '', availability: '12:00 PM' });
    window.location.reload();
  }


  return (
    <body>
      <Donor_Header />
      <section className="push"></section>
      <div className="row restaurant-cards"> {/* CARDS */}
        <div className="col card create-task mt-3 mx-4"> {/* FIRST CARD */}
          <FontAwesomeIcon icon={faClipboardCheck} size="2xl" />
          <p></p>
          <div className="card-body">
            <div className="row">
              <h5 className="card-title">CREATE NEW TASK</h5>
            </div>
            <form id="form" className="row" onSubmit={sendTask}>
              <label htmlFor="title"><b>Donation Title:</b></label>
              <input type="text" onChange={(e) => setFormValue({ ...formValue, title: e.target.value })} placeholder="Give a name for your donation." name="title" required />
              <label htmlFor="quantity"><b>Quantity:</b></label>
              <input type="text" onChange={(e) => setFormValue({ ...formValue, quantity: e.target.value })} placeholder="How many servings (roughly)?" name="quantity" required />
              <label htmlFor="description"><b>Description:</b></label>
              <input type="text" onChange={(e) => setFormValue({ ...formValue, description: e.target.value })} placeholder="Provide a brief description of what you're donating." name="description" required />
              <p> </p>
              <p> </p>
              <label htmlFor="availability">When can this donation be picked up?</label>
              <select name="availability" onChange={(e) => setFormValue({ ...formValue, availability: e.target.value })} id="availability">
                <option value="12PM">12:00 PM</option>
                <option value="1PM">1:00 PM</option>
                <option value="2PM">2:00 PM</option>
                <option value="3PM">3:00 PM</option>
                <option value="4PM">4:00 PM</option>
                <option value="5PM">5:00 PM</option>
                <option value="6PM">6:00 PM</option>
                <option value="7PM">7:00 PM</option>
                <option value="8PM">8:00 PM</option>
                <option value="9PM">9:00 PM</option>
                <option value="10PM">10:00 PM</option>
                <option value="11PM">11:00 PM</option>
                <option value="12AM">12:00 AM</option>
                <option value="1AM">1:00 AM</option>
                <option value="2AM">2:00 AM</option>
              </select>
              <p> </p>
              <p> </p>
              <button id="post" type="submit">POST</button>
            </form>
          </div>
        </div> {/*END FIRST CARD*/}
        <div className="col card upcoming-trips mt-3">

          <Donor_Timeline></Donor_Timeline>
        </div>

      </div>


    </body>
  )
}
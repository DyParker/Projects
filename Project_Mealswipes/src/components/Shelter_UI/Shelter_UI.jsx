import { useState, useEffect } from 'react';
import firebase from 'firebase/compat/app';
import 'firebase/compat/firestore';
import 'firebase/compat/auth';
import 'firebase/compat/analytics';
import SHeader from '../Shelter_Header/Shelter_Header.jsx';
import './Shelter_UI.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComment, faClock } from '@fortawesome/free-solid-svg-icons';
import Shelter_Active_Timeline from './Shelter_Active_Timeline.jsx';

export default function Shelter_UI() {
  const auth = firebase.auth();
  const db = firebase.firestore();
  const [status, setStatus] = useState('');

  const[curr_uid, setCurr_UID] = useState('')

    useEffect(() => {
        const unsub = auth.onAuthStateChanged((authObj) => {
            unsub();
            if (authObj) {
                setCurr_UID(authObj.uid);
            } else {
                // not logged in
            }
        });
    }, []);

  useEffect(() => {
    const userRef = auth.currentUser && db.collection('users').doc(auth.currentUser.uid);
  
    if (userRef) {
      userRef.get().then((doc) => {
        if (doc.exists) {
          const newStatus = doc.data().status || '';
          setStatus(newStatus);
        } else {
          console.log('No such document!');
        }
      });
    }
  }, [auth.currentUser]);

  const handleToggle = () => {
    const userRef = db.collection('users').doc(curr_uid);

    const newStatus = status === 'open' ? 'closed' : 'open';
    userRef.update({ status: newStatus }).then(() => {
      setStatus(newStatus);
    });
  };

  return (
    <body>
      <SHeader></SHeader>
      <section className="push"></section>

      <div className="row">
        <div className="col">
          <div className="d-flex justify-content-center my-3">
          <h6 className="toggle-text">{status === 'open' ? 'Accepting Tasks' : 'Closed'}</h6>
          </div>
          <div className="d-flex justify-content-center my-3">
            
            <label className="switch">
              <input type="checkbox" checked={status === 'open'} onChange={handleToggle} />
              <div className="slider round"></div>
            </label>
            
            
          </div>
        </div>
      </div>

      <div className="row shelter-cards">
        <div className="col card next-trips mt-3 mx-4">
          <FontAwesomeIcon icon={faClock} color="black" size="2xl" />
          <div className="card-body">
            <h5 className="card-title">UPCOMING DELIVERIES</h5>
            <Shelter_Active_Timeline></Shelter_Active_Timeline>
            
          </div>
        </div>
      </div>

    </body>
  )
}



/* import SHeader from '../Shelter_Header/Shelter_Header.jsx';
import './Shelter_UI.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComment, faClock } from '@fortawesome/free-solid-svg-icons';
import { useState } from 'react';
import firebase from 'firebase/compat/app';
import 'firebase/compat/firestore';
import 'firebase/compat/auth';
import 'firebase/compat/analytics';


export default function Shelter_UI() {

  const [status, setStatus] = useState('open');

  const updateStatus = () => {
    firebase.auth().onAuthStateChanged(function (user) {
      if (user) {
        const db = firebase.firestore();
        db.collection('users').doc(user.uid).update({ status: status })
          .then(() => {
            console.log('Status updated successfully!');
          })
          .catch((error) => {
            console.error('Error updating status: ', error);
          });
      } else {
        console.log('User is not authenticated!');
      }
    });
  }

  return (
    <body>
      <SHeader></SHeader>
      <section className="push"></section>
      <div className="col">
                <button onClick={() => setStatus(status === 'open' ? 'closed' : 'open')}>{status === 'open' ? 'Close' : 'Open'}</button>
              </div>


      <div className="row shelter-cards"> 
        <div className="col card next-trips mt-3 mx-4"> 
          <FontAwesomeIcon icon={faClock} color="black" size="2xl" />
          <div className="card-body">
            <h5 className="card-title">UPCOMING DELIVERIES</h5>
            <div className="row py-3 border-bottom"> 
              <div className="col">
                <div className="card-title my-0 mb-2 h6">Friday, March 17th</div>
                <p className="small m-0">Time: 2:30 pm</p>
              </div>
              <div className="col">
                <button><u>See details</u></button>
              </div>
              <div className="col">
                <button className="startchat"><a href="chat.html"><FontAwesomeIcon icon={faComment} color="black" size="2xl" /></a></button>
              </div>
            </div>
          </div>
        </div>
      </div>

    </body>
  )

}
*/
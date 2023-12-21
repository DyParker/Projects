import './Shelter_UI';
import { db, auth } from "../../firebase";
import { useCollectionData } from "react-firebase-hooks/firestore";
import { useCol } from "react-bootstrap/Col";
import { useEffect, useRef, useState } from "react";
import { collection, query, where, orderBy, limit, getDoc, updateDoc, doc } from "firebase/firestore";

import TaskConfirm from "./Task_Confirm.jsx";


import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComment, faMapLocationDot, faCheck } from '@fortawesome/free-solid-svg-icons';

import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';



function Tasks(props) {
        
    
    const { task } = props;
    const { v_uid, address, createdAt, description, status, uid, photoURL, quantity, title, availability, ref } = task;


    const [showModal, setShowModal] = useState(false);
    const [showConfirm, setShowConfirm] = useState(false);

    const [userName, setUserName] = useState("");

  useEffect(() => {
    const fetchUserName = async () => {
      try {
        const userDoc = await getDoc(doc(db, "users", v_uid));
        if (userDoc.exists()) {
          const userData = userDoc.data();
          setUserName(userData.name);
        }
      } catch (error) {
        console.error(error.message);
      }
    };

    fetchUserName();
  }, [v_uid]);

    


    const handleRejectClick = () => {
        setShowModal(false);
    };
    const modalStyle = {
        display: showModal ? 'left' : 'none',
        justifyContent: 'left',
        position: 'fixed',
        top: 100,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        zIndex: 999,
    };

    const minsAgo = (oldDate) => {
        const currentDate = new Date();
        const diff = (currentDate - oldDate) / 60000;
        let result = 0;
        if (diff > 60) {
            result += Math.floor(diff / 60);
            return result + " hours ago";
        }
        else return Math.floor(diff) + " minutes ago";
    }

    return (
        <>

            <div className="row py-3 border-bottom">
                <div className="col">
                    <div className="card-title my-0 mb-2 h6">{`Title: ${title}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Time: ${availability}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Quantity: ${quantity}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Delivered by: ${userName}`}</div>
                </div>
                <div className="col">
                    <button>{`Status: ${status}`}</button>
                </div>

                <div className="col">
                    <button onClick={() => setShowModal(true)}>Details:</button>
                </div>
                <div className="col buttons">
                    <button className="accept" onClick={() => setShowConfirm(true)}><FontAwesomeIcon icon={faCheck} color="white" size="2xl" /></button>
                </div>
            </div>
            <TaskConfirm show={showConfirm} onClose={()=>setShowConfirm(false)} taskRef = {ref}></TaskConfirm>
            <Modal
                show={showModal}
                onHide={handleRejectClick}
                backdrop="static"
                keyboard={false}
            >
                <Modal.Header closeButton>
                    <Modal.Title>Description</Modal.Title>
                </Modal.Header>
                <Modal.Footer>
                    <span className="taskDescription">{description}</span>
                    <Button variant="secondary" onClick={handleRejectClick}>
                        Close
                    </Button>
                </Modal.Footer>
            </Modal>
            
        </>
    );
}

export default function Shelter_Active_Timeline() {
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


    const dummy = useRef();
    const q = query(collection(db, "tasks"), where("status", "==", "in-progress"), where("s_uid", "==", curr_uid), orderBy("createdAt"),
        limit(25));

    const [tasks, loading, error] = useCollectionData(q, { idField: 'id' });

    if (error) {
        console.error(error.message);
    }




    return (
        <div>
            {tasks ? tasks && tasks.map(task => <Tasks key={task.id} task={task} />) : 'Loading...'}
            <span ref={dummy}></span>
        </div>
    )
}
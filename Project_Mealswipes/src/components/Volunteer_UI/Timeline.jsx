import './Volunteer_UI.css';
import { db } from "../../firebase";
import { useCollectionData } from "react-firebase-hooks/firestore";
import {useCol} from "react-bootstrap/Col";
import {useEffect, useRef, useState} from "react";
import { collection, query, where, orderBy, limit, getDocs, updateDoc, doc } from "firebase/firestore";


import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComment, faMapLocationDot, faCheck } from '@fortawesome/free-solid-svg-icons';
import ShelterSelect from "./ShelterSelect";

function Tasks(props){
    const { task } = props;
    const { address, createdAt, description, status, uid, photoURL, quantity, title, availability, ref, d_address } = task;

    const [showModal, setShowModal] = useState(false);

    const minsAgo = (oldDate) => {
        const currentDate = new Date();
        const diff= (currentDate-oldDate)/60000;
        let result = 0;
        if (diff>60){
            result += Math.floor(diff/60);
            return result + " hours ago";
        }
        else return Math.floor(diff) + " minutes ago";
    }

    let addressSetter = 'https://www.google.com/maps/dir/?api=1&destination=';
    let travelMode = '&travelmode=driving/';
    let destination = encodeURIComponent(d_address);
    let firstHalf = addressSetter.concat(destination);
    let mapLink = firstHalf.concat(travelMode);



    return (
        <>

            <div className="row py-3 border-bottom">
                <div className="col">
                    <div className="card-title my-0 mb-2 h6">{`Available at: ${availability}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Title: ${title}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Quantity: ${quantity}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Donor Address: ${d_address}`}</div>
                </div>
                <div className="col buttons">
                    <button className="viewmap"> <a href= {mapLink} target={'_blank'}><FontAwesomeIcon icon={faMapLocationDot} size="2xl" /></a></button>
                    <button className="accept" onClick={() => setShowModal(true)}><FontAwesomeIcon icon={faCheck} color="white" size="2xl" /></button>
                </div>
            </div>
            <ShelterSelect show={showModal} onClose={()=>setShowModal(false)} taskRef = {ref}></ShelterSelect>

        </>
    );
}


/*
<div className="row py-3 border-bottom">
                <div className="col">
                    <div className="card-title my-0 mb-2 h6">{`Posted: ${minsAgo(createdAt)}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Available at: ${availability}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Title: ${title}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Quantity: ${quantity}`}</div>
                </div>
                <div className="col buttons">
                    <button className='chatbubbles'><a href="chat.html"><FontAwesomeIcon icon={faComment} color="white" size="2xl" /></a></button>
                    <button className="viewmap"><FontAwesomeIcon icon={faMapLocationDot} size="2xl" /></button>
                    <button className="accept" onClick={() => setShowModal(true)}><FontAwesomeIcon icon={faCheck} color="white" size="2xl" /></button>
                </div>
            </div>
 */
export default function Timeline(){
    const dummy = useRef();
    const q = query(collection(db, "tasks"), where("status", "==", "available"), orderBy("createdAt"),
        limit(25));

    const [tasks, loading, error] = useCollectionData(q, { idField: 'id' });

    if (error) {
        console.error(error.message);
    }

    const [taskStatus, setTaskStatus ] = useState('available');

    function changeTaskStatus() { // handle clicking 'accept'!!!
        setTaskStatus('in-progress');
    }




    return(
        <>
        <div>
            {tasks ? tasks && tasks.map(task => <Tasks key={task.id} task={task} />) : 'Loading...'}
            <span ref={dummy}></span>
        </div>
        </>
    )
}
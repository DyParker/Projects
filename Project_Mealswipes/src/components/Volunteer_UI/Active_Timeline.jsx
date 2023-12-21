import './Volunteer_UI.css';
import { db, auth } from "../../firebase";
import { useCollectionData } from "react-firebase-hooks/firestore";
import { useCol } from "react-bootstrap/Col";
import { useEffect, useRef, useState } from "react";
import { collection, query, where, orderBy, limit, getDocs, updateDoc, doc } from "firebase/firestore";


import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComment, faMapLocationDot, faCheck } from '@fortawesome/free-solid-svg-icons';


function Tasks(props) {
    const { task } = props;
    const { s_address, createdAt, description, status, uid, photoURL, quantity, title, availability, ref } = task;

    let addressSetter = 'https://www.google.com/maps/dir/?api=1&destination=';
    let travelMode = '&travelmode=driving/';
    let destination = encodeURIComponent(s_address);
    let firstHalf = addressSetter.concat(destination);
    let mapLink = firstHalf.concat(travelMode);


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
                    <div className="card-title my-0 mb-2 h6">{`Available at: ${availability}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Title: ${title}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Quantity: ${quantity}`}</div>
                    <div className="card-title my-0 mb-2 h6">{`Deliver to: ${s_address}`}</div>
                </div>
                <div className="col buttons">
                    <button className="viewmap"> <a href={mapLink} target={'_blank'}><FontAwesomeIcon icon={faMapLocationDot} size="2xl" /></a></button>
                </div>
            </div>
        </>
    );
}

export default function Active_Timeline() {
    const [curr_uid, setCurr_UID] = useState('')

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
    const q = query(collection(db, "tasks"), where("status", "==", "in-progress"), where("v_uid", "==", curr_uid), orderBy("createdAt"),
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
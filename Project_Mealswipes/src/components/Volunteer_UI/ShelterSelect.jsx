



import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import { collection, doc, limit, orderBy, query, updateDoc, where } from "firebase/firestore";
import { auth, db } from "../../firebase";
import React, { useEffect, useState } from "react";
import { useCollectionData } from "react-firebase-hooks/firestore";
import { wait } from "@testing-library/user-event/dist/utils";





const ShelterOption = props => {
    const { shelter } = props;
    const { address, name, ref } = shelter;

    console.log(`name: ${name}, ref: ${ref}`);

    return (
        <option value={ref}>{`${name}, ${address}`}</option>
    )
}

const ShelterSelect = props => {
    const closeOnEscapeKeyDown = (e) => {
        if ((e.charCode || e.keyCode) === 27) {
            props.onClose()
        }
    }
    useEffect(() => {

        document.body.addEventListener('keydown', closeOnEscapeKeyDown)
        return function cleanup() {
            document.body.removeEventListener('keydown', closeOnEscapeKeyDown)
        }
    }, [])

    const [choice, setChoice] = useState(null);

    const q = query(collection(db, "users"), where("role", "==", "shelter"), where("status", "==", "open"),
        limit(25));

    const [shelters, loading, error] = useCollectionData(q, { idField: 'id' });

    if (!props.show) return null;


    async function handleAccept(e) {
        e.preventDefault();
        const newTaskRef = doc(db, "tasks", props.taskRef);

        // Find the selected shelter's document in the "shelters" array
        const selectedShelter = shelters.find(shelter => shelter.ref === choice);

        // Extract the address from the selected shelter's document
        const s_address = selectedShelter.address;

        const data = {
            status: "in-progress",
            s_address: s_address,
            v_uid: auth.currentUser.uid,
            s_uid: choice
        }

        props.onClose();

        await updateDoc(newTaskRef, data)
            .then(() => {
                console.log("Updated)")
            })
            .catch(error => {
                console.log(error);
            })
    }

    return (
        <Modal
            show={props.show}
            onHide={props.onClose}
            backdrop="static"
            keyboard={false}
            onClick={e => {
                e.stopPropagation();
            }}
        >
            <Modal.Header>
                <Modal.Title>Where to Deliver?</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <select name="shelter" onChange={(e) => setChoice(e.target.value)} id="shelter" defaultValue={"default"} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }} required={true}>
                    <option value="default" disabled>Please Pick a Shelter</option>
                    {shelters && shelters.map(shelter => <ShelterOption key={shelter.id} shelter={shelter} />)}
                </select>
            </Modal.Body>
            <Modal.Footer>
                <Button onClick={choice ? handleAccept : null} variant="primary" type="submit" style={{ backgroundColor: '#87B692', borderColor: '#87B692' }}>
                    Accept
                </Button>
                <Button variant="secondary" onClick={props.onClose} style={{ backgroundColor: 'crimson', borderColor: 'crimson' }}>
                    Cancel
                </Button>
            </Modal.Footer>
        </Modal>
    )
}

export default ShelterSelect





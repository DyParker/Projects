import Modal from "react-bootstrap/Modal";
import Button from "react-bootstrap/Button";
import React, {useEffect, useState} from "react";
import firebase from "firebase/compat/app";







const ChangePassword = props => {
    const closeOnEscapeKeyDown = (e) =>{
        if((e.charCode || e.keyCode) === 27){
            setCurrentPassword('');
            setNewPassword('');
            setConfirmPassword('');
            props.onClose()
        }
    }
    useEffect(() => {
        document.body.addEventListener('keydown', closeOnEscapeKeyDown)
        return function cleanup(){
            document.body.removeEventListener('keydown', closeOnEscapeKeyDown)
        }
    }, [])

    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');

    if(!props.show) return null;


    async function handleSubmit (e) {
        e.preventDefault();
        if (newPassword === confirmPassword) {
            const user = firebase.auth().currentUser;
            const credential = firebase.auth.EmailAuthProvider.credential(
                user.email,
                currentPassword
            );

            user
                .reauthenticateWithCredential(credential)
                .then(() => {
                    user.updatePassword(newPassword).then(() => {
                        console.log('Password updated successfully');
                    });
                })
                .catch((error) => {
                    console.error(error);
                });
        } else {
            console.error('New password and confirm password do not match');
        }
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
        props.onClose();

    }

    return(
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
                <Modal.Title>Change Password</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <form onSubmit={handleSubmit}>
                    <label>
                        Current Password:
                        <input
                            type="password"
                            value={currentPassword}
                            onChange={(e) => setCurrentPassword(e.target.value)}
                            required
                        />
                    </label>
                    <label>
                        New Password:
                        <input
                            type="password"
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            required
                        />
                    </label>
                    <label>
                        Confirm New Password:
                        <input
                            type="password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            required
                        />
                    </label>
                    <button type="submit">Submit</button>
                </form>
                <p></p>
                <Button variant="secondary" onClick={props.onClose} style={{ backgroundColor: 'crimson', borderColor: 'crimson' }}>
                    Cancel
                </Button>
            </Modal.Body>
        </Modal>
    )
}

export default ChangePassword
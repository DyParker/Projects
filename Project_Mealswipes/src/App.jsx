import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from './components/Home/Home.jsx'
import About from './components/About/About.jsx';
import Sign_In from './components/Sign_In/Sign_In.jsx';
import Onboarding from './components/Onboarding/Onboarding.jsx';
import Volunteer_Sign_Up from './components/Volunteer_Sign_Up/Volunteer_Sign_Up.jsx';
import Volunteer_UI from './components/Volunteer_UI/Volunteer_UI.jsx';
import V_Profile_Info from './components/V_Profile_Info/V_Profile_Info.jsx';
import V_Reviews from './components/V_Reviews/V_Reviews.jsx';
import Donor_Sign_Up from './components/Donor_Sign_Up/Donor_Sign_Up.jsx';
import Shelter_Sign_Up from './components/Shelter_Sign_Up/Shelter_Sign_Up.jsx';
import Donor_UI from './components/Donor_UI/Donor_UI.jsx';
import Donor_Info from './components/Donor_Info/Donor_Info.jsx';
import Donor_Reviews from './components/Donor_Reviews/Donor_Reviews.jsx';
import Donor_History from "./components/Donor_History/Donor_History.jsx";
import Shelter_UI from './components/Shelter_UI/Shelter_UI.jsx';
import Shelter_Info from './components/Shelter_Info/Shelter_Info.jsx';
import Shelter_Reviews from './components/Shelter_Reviews/Shelter_Reviews.jsx';
import A_Home from './components/A_Home/A_Home.jsx';
import A_About from './components/A_About/A_About.jsx';


import React from "react";

import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import 'bootstrap/dist/js/bootstrap';



function App() {

  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/about' element={<About />} />
        <Route path='/a_home' element={<A_Home />} />
        <Route path='/a_about' element={<A_About />} />
        <Route path='/signin' element={<Sign_In />} />
        <Route path='/onboarding' element={<Onboarding />} />
        <Route path='/volsignup' element={<Volunteer_Sign_Up />} />
        <Route path='/volunteerui' element={<Volunteer_UI />} />
        <Route path='/vprofinfo' element={<V_Profile_Info />} />
        <Route path='/vreviews' element={<V_Reviews />} />
        <Route path='/donsignup' element={<Donor_Sign_Up />} />
        <Route path='/shelsignup' element={<Shelter_Sign_Up />} />
        <Route path='/donorui' element={<Donor_UI />} />
        <Route path='/donorinfo' element={<Donor_Info />} />
        <Route path='/donorreviews' element={<Donor_Reviews />} />
        <Route path='/donorhistory' element={<Donor_History />} />
        <Route path='/shelui' element={<Shelter_UI />} />
        <Route path='/shelinfo' element={<Shelter_Info />} />
        <Route path='/shelreviews' element={<Shelter_Reviews />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App;

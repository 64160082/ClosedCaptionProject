-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jul 31, 2024 at 03:39 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `flask_users`
--

-- --------------------------------------------------------

--
-- Table structure for table `audiofiles`
--

CREATE TABLE `audiofiles` (
  `audio_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `file_name` varchar(255) DEFAULT NULL,
  `file_type` varchar(10) DEFAULT NULL,
  `file_size` int(11) DEFAULT NULL,
  `upload_datetime` timestamp NOT NULL DEFAULT current_timestamp(),
  `audio_content` longblob DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `jsonfiles`
--

CREATE TABLE `jsonfiles` (
  `json_id` int(11) NOT NULL,
  `audio_id` int(11) NOT NULL,
  `file_name` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `json_content` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`json_content`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `user_id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `vttfiles`
--

CREATE TABLE `vttfiles` (
  `vtt_id` int(11) NOT NULL,
  `audio_id` int(11) DEFAULT NULL,
  `file_name` varchar(255) DEFAULT NULL,
  `vtt_content` longtext DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `audiofiles`
--
ALTER TABLE `audiofiles`
  ADD PRIMARY KEY (`audio_id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `jsonfiles`
--
ALTER TABLE `jsonfiles`
  ADD PRIMARY KEY (`json_id`),
  ADD KEY `audio_id` (`audio_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`user_id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `vttfiles`
--
ALTER TABLE `vttfiles`
  ADD PRIMARY KEY (`vtt_id`),
  ADD KEY `audio_id` (`audio_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `audiofiles`
--
ALTER TABLE `audiofiles`
  MODIFY `audio_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `jsonfiles`
--
ALTER TABLE `jsonfiles`
  MODIFY `json_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `user_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `vttfiles`
--
ALTER TABLE `vttfiles`
  MODIFY `vtt_id` int(11) NOT NULL AUTO_INCREMENT;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `audiofiles`
--
ALTER TABLE `audiofiles`
  ADD CONSTRAINT `audiofiles_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`user_id`) ON DELETE CASCADE;

--
-- Constraints for table `jsonfiles`
--
ALTER TABLE `jsonfiles`
  ADD CONSTRAINT `jsonfiles_ibfk_1` FOREIGN KEY (`audio_id`) REFERENCES `audiofiles` (`audio_id`) ON DELETE CASCADE;

--
-- Constraints for table `vttfiles`
--
ALTER TABLE `vttfiles`
  ADD CONSTRAINT `vttfiles_ibfk_1` FOREIGN KEY (`audio_id`) REFERENCES `audiofiles` (`audio_id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;

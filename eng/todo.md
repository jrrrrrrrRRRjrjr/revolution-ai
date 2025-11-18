1. Create the Home Screen (Service introduction & Start button)
2. Implement Login/Sign-up functionality
3. Create a Mode Selection Screen (In a Relationship / Breakup)
4. Build Required Input Fields (add one at a time, not all at once):

   * Age (User/Partner)
   * Gender (User/Partner)
   * MBTI (User/Partner)
   * Duration of Relationship (Talking stage + Dating period)
   * Long-distance relationship (Yes/No)
   * Attach KakaoTalk text file
5. Create an input field where the user can type their question
6. Send the user’s question, input data, and prompt to the API
7. Receive and display the API’s response
8. Save user records

**Database Design**

1. **User Information:** Name, Email, Password, Subscription Plan, Payment ID
2. **Generated Answer Files**
3. **Columns, Data Type, Description (Hobby — to be implemented later)**

   * `hobby_id`, Integer (Primary Key), Unique ID for each hobby
   * `user_id`, Integer (Foreign Key), References `user_id` in the Users table
   * `hobby_name`, String, e.g., “Screen Baseball,” “Gaming,” “Karaoke”
   * `category`, String, e.g., “Sports,” “Indoor,” “Instant Relief”

from Voice_Cloning.command_line_interface import VoiceCloner

voiceCloner = VoiceCloner()
#voiceCloner.play()
voiceCloner.synthesize("My first experience of segregation was back when i was a little boy, i believe around the age of six.\n I was playing with my friends, who happened to be white and their father came out and declared that they were no longer to play with me or be my friend because I was black.\n I was unaware of the connotation of my color would bring me until that point in time.")
voiceCloner.vocode()
print('done!')
